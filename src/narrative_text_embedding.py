# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:39:42 2022

@author: 16028
"""


import os
import argparse
import torch
from transformers import BertModel, BertConfig, BertTokenizer
import time
from nltk import word_tokenize
import numpy as np


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


dftest = pd.read_csv('https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-ak-tmax-inv-recent-v1.0.0-20220406', 
                     delimiter=' ', names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])


dftest = pd.read_csv('https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-ak-tmin-inv-recent-v1.0.0-20220406')
dftest.head()


dftest4 = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\climate_data_ex.txt', 
                      delim_whitespace=True,  names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])

dftest4['col1']
dftest4
df = df2019.copy()


df2 = pd.read_csv(r'C:\Users\16028\Downloads\Noce-etal_2019.tab', delimiter='\t')

sent_concat_train = []
y_train = []

for i in range(len(word_dict_train)):
    sent_concat_train.append(" ".join(word_dict_train[i]['sentence1_words']))
    sent_concat_train.append(" ".join(word_dict_train[i]['sentence2_words']))

    y_train.append(word_dict_train[i]['label'])
    


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased", output_hidden_states = True)
encoded_input = tokenizer(sent_concat_train, padding=True, truncation=True,return_tensors='pt') # padding="max_length", truncation=True,

start = time.time()
N = len(sent_concat_train)
i = 0
inc_num = 5
ii = []
tti = []
am = []
hidden_states = []

with torch.no_grad():
    while i < len(sent_concat_train):
        if i%1000 == 0: print(i)
        ii = encoded_input['input_ids'][i:i+inc_num]
        tti = encoded_input['token_type_ids'][i:i+inc_num]
        am = encoded_input['attention_mask'][i:i+inc_num]

        output=model(input_ids=ii, attention_mask=torch.tensor(am), token_type_ids=tti)
        hidden_states.append(output.last_hidden_state)
        i = i + inc_num 
        # 0:5->5:10->10:15 ->45:50->50:55 (i think it's working?')
        

end = time.time()
print(round(end-start,1))
###################################################################################

total_list = []
for tens in hidden_states:
    for i in tens:
        srmw = torch.mean(i, dim=0) 
        total_list.append(srmw)

sent1_embeds_train = []
sent2_embeds_train = []

for i in range(len(total_list)):
    if i%2 == 0: 
        sent1_embeds_train.append(total_list[i])
    else:
        sent2_embeds_train.append(total_list[i])
        
both_list_train = []
for i in range(len(sent1_embeds_train)):
    both_list_train.append([sent1_embeds_train[i], sent2_embeds_train[i]])
    
