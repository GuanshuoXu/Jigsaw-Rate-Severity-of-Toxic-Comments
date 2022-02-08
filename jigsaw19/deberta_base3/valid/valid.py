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

class JRSDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_len):
        self.text_list=text_list
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __len__(self):
        return len(self.text_list)
    def __getitem__(self, index):
        tokenized = self.tokenizer(text=self.text_list[index],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_tensors='pt')
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

class JRSModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super(JRSModel, self).__init__(config)
        self.deberta = DebertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, 7)
        self.init_weights()
    @autocast()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state'][:,0,:]
        logits = self.classifier(outputs)
        return logits

def main():

    start_time = time.time()

    ###
    df = pd.read_csv('../../../input/validation_data.csv')
    more_toxic_list = df['more_toxic'].values
    less_toxic_list = df['less_toxic'].values
    print(len(more_toxic_list), len(less_toxic_list))

    # parameters
    max_len = 256
    batch_size = 96
    model_path = "microsoft/deberta-base"

    # build model
    more_toxic_pred = np.zeros((len(more_toxic_list), 7), dtype=np.float32)
    less_toxic_pred = np.zeros((len(less_toxic_list), 7), dtype=np.float32)

    config = DebertaConfig.from_pretrained(model_path)
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../train/weights/weights', config=config)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # iterator for validation
    dataset = JRSDataset(more_toxic_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=8, 
                           pin_memory=True)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            more_toxic_pred[start:end] += logits.sigmoid().cpu().data.numpy()

    # iterator for validation
    dataset = JRSDataset(less_toxic_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=8, 
                           pin_memory=True)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            less_toxic_pred[start:end] += logits.sigmoid().cpu().data.numpy()

    ###
    less_toxic_score = np.sum(less_toxic_pred, axis=1)
    more_toxic_score = np.sum(more_toxic_pred, axis=1)
    print(np.mean(less_toxic_score<more_toxic_score))

    ###
    from geneticalgorithm import geneticalgorithm as ga
    def loss_func(wt):
        less_toxic_score = np.zeros((len(less_toxic_pred), ), dtype=np.float16)
        more_toxic_score = np.zeros((len(more_toxic_pred), ), dtype=np.float16)
        for i in range(7):
            less_toxic_score += less_toxic_pred[:,i]*wt[i]
            more_toxic_score += more_toxic_pred[:,i]*wt[i]
        return 1.0 - np.mean(less_toxic_score<more_toxic_score)

    varbound=np.array([[0,20]]*7)

    model=ga(function=loss_func, dimension=7, variable_type='int', variable_boundaries=varbound)

    model.run()
    best_wt=model.output_dict['variable']

    ### validate result again
    less_toxic_score = np.zeros((len(less_toxic_pred), ), dtype=np.float16)
    more_toxic_score = np.zeros((len(more_toxic_pred), ), dtype=np.float16)
    for i in range(7):
        less_toxic_score += less_toxic_pred[:,i]*best_wt[i]
        more_toxic_score += more_toxic_pred[:,i]*best_wt[i]
    print(np.mean(less_toxic_score<more_toxic_score))

    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
