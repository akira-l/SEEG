#!/usr/bin/python

import os 
from tqdm import tqdm 

import pdb 

candi_list = [18, 406, 190, 455, 389, 299, 190, 455, 389, 554, 12, 48, 435, 319, 320, 360, 227, 
        271, 486, 132, 21, 32, 40, 49, 110, 131, 140, 154, 236, 662, 638, 16, 29, 30, 40, 70, 115, 129, 142, 188] 
vid_list = [] 
time_list = [] 

#vid_list = ['d38LKbYfWrs', 'awADEuv5vWY', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'B905LapVP7I', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'gVfgkFaswn4', 'LujWrkYsl64', 'GMynksvCcUI', '1mLQFm3wEfw', 'fWqKalpYgLo', 'YATYsgi3e5A', '7MHOk7qVhYs', 'uTbA-mxo858', 'HR9956gDpUY', 'YUUP2MMz7PU', '8Z24LCysq3A', '1N39Z0ODeME', '1L6l-FiV4xo', 'Gn2W3X_pGh4', 'NCJTV5KaJJc', 'Fivy99RtMfM', 'LFJ9WAHowcg', 'fbAj9JfCXng', 'IWjzT2l5C34', 'd6K-sePuFAo', '5CSDIcUsIJk', 'LZXUR4z2P9w', 'MgnnQ2CN6yY', 'FV-c2FnPnDE', 'h27g5iT0tck', 'Gn2W3X_pGh4', 'N8Votwxx8a0', 'MgOVOCUuScE', 'fm_0sTNcDIo', 'h9SKyrHRhDo', '-Z-ul0GzzM4']

handle = open('train_save.txt') 
cont = handle.readlines() 

count_bar = tqdm(total=len(candi_list)) 
for cand in candi_list: 
    count_bar.update(1) 
    sub_cont = cont[cand - 1]
    sub_split = sub_cont.split('  ') 
    try: 
        assert str(cand) in sub_split 
    except: 
        pdb.set_trace()
    for s_cont in sub_split: 
        try: 
            s_cont[0] 
        except: 
            pdb.set_trace() 
        if s_cont[0] == '(': 
            vid_info = s_cont[1:-2] 
            vid = vid_info.split(',')[0] 
            time_seq = vid_info.split(' ')[-1] 
            time_list.append(time_seq) 
            vid_list.append(vid) 
            break 
count_bar.close() 
print(vid_list) 
print(time_list) 



