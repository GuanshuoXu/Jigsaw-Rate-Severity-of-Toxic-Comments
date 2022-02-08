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
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

class JRSRobertaBaseDataset(Dataset):
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
                                   return_tensors='pt')
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze()

class JRSRobertaBaseModel(BertPreTrainedModel):
    def __init__(self, config):
        super(JRSRobertaBaseModel, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
    @autocast()
    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        classification_output = outputs[1]
        logits = self.classifier(classification_output)
        return logits

def main():

    start_time = time.time()

    ###
    df = pd.read_csv('../../../input/validation_data.csv')
    more_toxic_list = df['more_toxic'].values
    less_toxic_list = df['less_toxic'].values
    print(len(more_toxic_list), len(less_toxic_list))

    # parameters
    max_len = 192
    batch_size = 96
    model_path = "roberta-base"

    # build model
    more_toxic_pred = np.zeros((len(more_toxic_list), ), dtype=np.float32)
    less_toxic_pred = np.zeros((len(less_toxic_list), ), dtype=np.float32)

    config = RobertaConfig.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSRobertaBaseModel.from_pretrained('../train/weights/weights', config=config)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # iterator for validation
    dataset = JRSRobertaBaseDataset(more_toxic_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=8, 
                           pin_memory=True)

    for j, (batch_input_ids, batch_attention_mask) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask).view(-1)
            more_toxic_pred[start:end] += logits.sigmoid().cpu().data.numpy()

    # iterator for validation
    dataset = JRSRobertaBaseDataset(less_toxic_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=8, 
                           pin_memory=True)

    for j, (batch_input_ids, batch_attention_mask) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask).view(-1)
            less_toxic_pred[start:end] += logits.sigmoid().cpu().data.numpy()

    ###
    print(less_toxic_pred.shape, more_toxic_pred.shape)
    print(np.mean(less_toxic_pred<more_toxic_pred))

    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
