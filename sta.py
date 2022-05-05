#!/usr/bin/python


import os 

import collections
import nltk 
from nltk import data
data.path.append('./') 
#nltk.download('punkt') 
#nltk.download('averaged_perceptron_tagger') 

from tqdm import tqdm 

import torch 

import pdb 

train_sents = torch.load('train_words.pt') 
val_sents = torch.load('val_words.pt') 

all_sents = train_sents + val_sents 

all_words = [] 
word_table = {} 
count_bar = tqdm(total=len(all_sents)) 
for sent in all_sents: 
    count_bar.update(1) 
    tags = nltk.pos_tag(nltk.word_tokenize(sent))
    tags_dict = {} 
    for tmp_tag in tags: 
        tags_dict[tmp_tag[0]] = tmp_tag[1]

    split_sent = sent.split(' ') 
    for tmp_word in split_sent: 
        if tmp_word in tags_dict: 
            cur_tag = tags_dict[tmp_word] 
        else: 
            cur_tag = 'None'

        if tmp_word in word_table: 
            word_table[tmp_word].append(cur_tag) 
        else: 
            word_table[tmp_word] = [cur_tag] 

    all_words.extend(split_sent) 
            
count_bar.close() 

words_count = collections.Counter(all_words) 
rerank_count = {k: v for k, v in sorted(words_count.items(), key=lambda item: item[1])}
#rerank_count = dict(sorted(words_count.items(), key=lambda item: item[1]))

count_bar = tqdm(total=len(rerank_count))
handle = open('words_tag.txt', 'a') 
for sub_item in rerank_count.items(): 
    count_bar.update(1) 
    sub_tags = word_table[sub_item[0]] 
    tag_count = collections.Counter(sub_tags) 
    retag_count = {k: v for k,v in sorted(tag_count.items(), key=lambda item: item[1])} 
    max_tag = list(retag_count.keys())[-1] 

    if 'JJ' in max_tag or 'NN' in max_tag or 'RB' in max_tag or 'V' in max_tag: 
        cont = sub_item[0] + ' ' + str(sub_item[1]) + ' ' + max_tag + '\n'
        handle.write(cont) 
handle.close() 
count_bar.close() 

pdb.set_trace()



