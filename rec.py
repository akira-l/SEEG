#!/usr/bin/python

import os 
import torch 

from tqdm import tqdm 

import pdb 

train_gather = torch.load('train_gather_dict.pt') 


handle = open('vid_list.txt', 'a') 
count_bar = tqdm(total=len(train_gather))
for sub_key in train_gather.keys(): 
    count_bar.update(1) 
    meta = train_gather[sub_key] 
    text = meta['word_list'] 
    vid = meta['vid'] 
    st_time = meta['start_time']
    en_time = meta['end_time'] 

    cont = ' '.join([vid, str(st_time), str(en_time), text])  
    handle.write(cont + '\n') 
count_bar.close() 
handle.close() 


