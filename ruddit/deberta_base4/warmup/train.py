import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import random
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
from transformers.models.deberta.modeling_deberta import ContextPooler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class JRSDebertaDataset(Dataset):
    def __init__(self, id_list, tokenizer, data_dict, max_len):
        self.id_list=id_list
        self.tokenizer=tokenizer
        self.data_dict=data_dict
        self.max_len=max_len
    def __len__(self):
        return len(self.id_list)
    def __getitem__(self, index):
        tokenized = self.tokenizer(text=self.data_dict[self.id_list[index]]['text'],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_tensors='pt')
        target = self.data_dict[self.id_list[index]]['labels']
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target

class JRSDebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super(JRSDebertaModel, self).__init__(config)
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = nn.Linear(output_dim, 1)
        self.init_weights()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        logits = self.classifier(pooled_output)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 8554
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    import pickle
    with open('../../splits/split1/train_id_list1.pickle', 'rb') as f:
        id_list = pickle.load(f)
    with open('../../splits/split1/data_dict.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    print(len(id_list), len(data_dict))

    # hyperparameters
    learning_rate = 0.00003
    max_len = 192
    batch_size = 32
    num_epoch = 3
    model_path = "microsoft/deberta-base"

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    config = DebertaConfig.from_pretrained(model_path)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSDebertaModel.from_pretrained(model_path, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    num_train_steps = int(len(id_list)/(batch_size*3)*num_epoch)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # training
    train_datagen = JRSDebertaDataset(id_list, tokenizer, data_dict, max_len)
    train_sampler = DistributedSampler(train_datagen)
    train_generator = DataLoader(dataset=train_datagen,
                                 sampler=train_sampler,
                                 batch_size=batch_size,
                                 num_workers=8,
                                 pin_memory=True)

    if args.local_rank == 0:
        start_time = time.time()

    scaler = GradScaler()
    for ep in range(num_epoch):
        losses = AverageMeter()
        model.train()
        for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_target) in enumerate(train_generator):
            batch_input_ids = batch_input_ids.to(args.device)
            batch_attention_mask = batch_attention_mask.to(args.device)
            batch_token_type_ids = batch_token_type_ids.to(args.device)
            batch_target = torch.from_numpy(np.array(batch_target)).float().to(args.device)

            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                loss = nn.BCEWithLogitsLoss()(logits.view(-1), batch_target) 

            losses.update(loss.item(), logits.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            #if args.local_rank == 0:
            #    print('\r',end='',flush=True)
            #    message = '%s %5.1f %6.1f %0.8f    |     %0.3f     |' % ("train",j/len(train_generator)+ep,ep,scheduler.get_lr()[0],losses.avg)
            #    print(message , end='',flush=True)

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    if args.local_rank == 0:
        out_dir = 'weights/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.module.state_dict(), out_dir+'weights')

    if args.local_rank == 0:
        end_time = time.time()
        print(end_time-start_time)

if __name__ == "__main__":
    main()
