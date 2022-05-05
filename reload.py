#!/usr/bin/python

import datetime
import logging
import os
import pickle
import random
import pyarrow

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm 

import pdb 

data_path = 'data/ted_dataset/lmdb_train_cache' 

src_lmdb_env = lmdb.open(data_path, readonly=True, lock=False) 
with src_lmdb_env.begin() as txn: 
    n_videos = txn.stat()['entries'] 

candi_list = [18, 406, 190, 455, 389, 299, 190, 455, 389, 554, 12, 48, 435, 319, 320, 360, 227, 
        271, 486, 132, 21, 32, 40, 49, 110, 131, 140, 154, 236, 662, 638, 16, 29, 30, 40, 70, 115, 129, 142, 188] 

vid_list = ['d38LKbYfWrs', 'awADEuv5vWY', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'B905LapVP7I', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'gVfgkFaswn4', 'LujWrkYsl64', 'GMynksvCcUI', '1mLQFm3wEfw', 'fWqKalpYgLo', 'YATYsgi3e5A', '7MHOk7qVhYs', 'uTbA-mxo858', 'HR9956gDpUY', 'YUUP2MMz7PU', '8Z24LCysq3A', '1N39Z0ODeME', '1L6l-FiV4xo', 'Gn2W3X_pGh4', 'NCJTV5KaJJc', 'Fivy99RtMfM', 'LFJ9WAHowcg', 'fbAj9JfCXng', 'IWjzT2l5C34', 'd6K-sePuFAo', '5CSDIcUsIJk', 'LZXUR4z2P9w', 'MgnnQ2CN6yY', 'FV-c2FnPnDE', 'h27g5iT0tck', 'Gn2W3X_pGh4', 'N8Votwxx8a0', 'MgOVOCUuScE', 'fm_0sTNcDIo', 'h9SKyrHRhDo', '-Z-ul0GzzM4']

time_list = ['0:08:45.677978-0:08:53.844173', '0:13:45.333333-0:13:50.333333', '0:03:16.125000-0:03:21.458333', '0:04:31.958333-0:04:40.208333', '0:00:22.208333-0:00:30.416667', '0:03:22.833333-0:03:32.208333', '0:03:16.125000-0:03:21.458333', '0:04:31.958333-0:04:40.208333', '0:00:22.208333-0:00:30.416667', '0:17:45.166667-0:17:56.791667', '0:10:18.876255-0:10:27.009276', '0:06:55.625000-0:07:02.291667', '0:00:52.360000-0:01:00', '0:02:51.400000-0:03:03.200000', '0:02:41.916667-0:02:51.583333', '0:02:43.500000-0:02:55', '0:01:35.440000-0:01:44.120000', '0:04:35.708333-0:04:43.500000', '0:06:28.958333-0:06:40.875000', '0:05:18.820538-0:05:29.859864', '0:06:19.633333-0:06:30.333333', '0:01:08.675853-0:01:16.075406', '0:03:50.708333-0:03:57.708333', '0:04:38.343536-0:04:48.592377', '0:04:08.800000-0:04:20.120000', '0:09:47.933910-0:09:55.141944', '0:03:25.708333-0:03:32.916667', '0:13:51.840000-0:14:02.120000', '0:05:25.916667-0:05:35.458333', '0:05:58.958333-0:06:08.833333', '0:04:21.375000-0:04:32.625000', '0:10:49.300000-0:10:55.866667', '0:04:39.695478-0:04:49.778348', '0:09:01.250000-0:09:08.416667', '0:03:50.708333-0:03:57.708333', '0:11:37.333333-0:11:46.291667', '0:00:40.366974-0:00:46.074173', '0:05:35.733583-0:05:42.066606', '0:16:19.600000-0:16:27.333333', '0:10:13.056902-0:10:20.723239']

gather_dict = {}  
regather = [] 
cur_gather = [] 
sec_gather = [] 

count_bar = tqdm(total=n_videos) 
with src_lmdb_env.begin(write=False) as txn: 

    for idx in range(n_videos): 
        count_bar.update(1) 
        key = '{:010}'.format(idx).encode('ascii')
        sample = txn.get(key)
    
        sample = pyarrow.deserialize(sample)
        word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample

        vid = aux_info['vid'] 

        if vid in vid_list: 
            time_key = time_list[vid_list.index(vid)]
            candi = candi_list[vid_list.index(vid)] 
            start_key = time_key.split('-')[0]  
            start_time_list = start_key.split(':') 
            end_key = time_key.split('-')[1]
            end_time_list = end_key.split(':') 

            sec_time = float(start_time_list[1])*60 + float(start_time_list[2]) 
            cur_start_time = aux_info['start_time'] 

            sec_end_time = float(end_time_list[1])*60 + float(end_time_list[2]) 
            cur_end_time = aux_info['end_time'] 

            if candi == 662: 
                cur_gather.append([cur_start_time, cur_end_time])
                sec_gather.append([sec_time, sec_end_time]) 

            if abs(cur_start_time - sec_time) < 3 and abs(cur_end_time - sec_end_time) < 3: 
                
                gather_dict[candi] = {} 
                gather_dict[candi]['aux_info'] = aux_info
                gather_dict[candi]['audio'] = audio 
                gather_dict[candi]['vec_seq'] = vec_seq 
                gather_dict[candi]['pose_seq'] = pose_seq 
                gather_dict[candi]['word_seq'] = word_seq 
                gather_dict[candi]['need_time'] = time_key 
                regather.append(candi) 

count_bar.close() 
pdb.set_trace() 

