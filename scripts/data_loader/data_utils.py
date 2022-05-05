#!/usr/bin/python

import datetime
import logging
import os
import pickle
import random

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModel

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import librosa

import pdb 


def proc_time_seq(feat, aux_info, feat_size = 128): 

    start_time = aux_info['start_time'] 
    end_time = aux_info['end_time'] 
    if 'word_seq' in aux_info: 
        word_seq = aux_info['word_seq'] 
    else: 
        word_seq = aux_info['words'] 
    duration = end_time - start_time 

    input_feat_size = feat.size() 
    assert len(input_feat_size) == 3
    init_feat = torch.zeros(feat_size, input_feat_size[0], input_feat_size[2])  

    assert input_feat_size[1] == len(word_seq) 

    for sub_word_item, sub_feat in zip(word_seq, feat.transpose(0,1)): 
        sub_start = max(start_time, sub_word_item[1]) 
        sub_end = min(end_time, sub_word_item[2]) 

        sub_dur = sub_start - sub_end
        dur_ind = (sub_dur / duration) * feat_size 
        dur_ind = int(dur_ind) 

        start_ind = (sub_start - start_time) / duration * feat_size 
        start_ind = int(start_ind) 
        end_ind = min(start_ind + dur_ind, feat_size - 1) 

        init_feat[start_ind:end_ind] = sub_feat 

    return init_feat 

        







