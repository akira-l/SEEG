import pprint
import time
from pathlib import Path
import sys

[sys.path.append(i) for i in ['.', '..']]

import matplotlib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import speech2gesture, vocab
from model.embedding_net import EmbeddingNet
from model.seq2seq_net import Seq2SeqNet
from train_eval.train_gan import train_iter_gan
from train_eval.train_joint_embed import train_iter_embed, eval_embed
from train_eval.train_seq2seq import train_iter_seq2seq
from train_eval.train_speech2gesture import train_iter_speech2gesture
from utils.average_meter import AverageMeter
from utils.data_utils import convert_dir_vec_to_pose
from utils.vocab_utils import build_vocab

matplotlib.use('Agg')  # we don't use interactive GUI

from config.parse_args import parse_args
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator, BeatGenerator

from torch import optim

from data_loader.lmdb_data_loader import *
import utils.train_utils
from utils.train_utils import create_video_and_save_single, set_logger

import librosa 
from tqdm import tqdm 

import pdb 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(args, lang_model, speaker_model, pose_dim, _device):
    # init model
    n_frames = args.n_poses
    generator = discriminator = loss_fn = None
    if args.model == 'multimodal_context':
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim).to(_device)
    elif args.model == 'joint_embedding':
        generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                 lang_model.word_embedding_weights, mode='random').to(_device)
    elif args.model == 'gesture_autoencoder':
        generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                 lang_model.word_embedding_weights, mode='pose').to(_device)
    elif args.model == 'seq2seq':
        generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                               lang_model.word_embedding_weights).to(_device)
        loss_fn = torch.nn.L1Loss()
    elif args.model == 'speech2gesture':
        generator = speech2gesture.Generator(n_frames, pose_dim, args.n_pre_poses).to(_device)
        discriminator = speech2gesture.Discriminator(pose_dim).to(_device)
        loss_fn = torch.nn.L1Loss()

    return generator, discriminator, loss_fn


def train_epochs(args, train_data_loader, val_data_loader, test_data_loader, lang_model, mean_dir_vec, pose_dim, speaker_model=None):

    
    #vid_list = ['d38LKbYfWrs', 'awADEuv5vWY', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'B905LapVP7I', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'gVfgkFaswn4', 'LujWrkYsl64', 'GMynksvCcUI', '1mLQFm3wEfw', 'fWqKalpYgLo', 'YATYsgi3e5A', '7MHOk7qVhYs', 'uTbA-mxo858', 'HR9956gDpUY', 'YUUP2MMz7PU', '8Z24LCysq3A', '1N39Z0ODeME', '1L6l-FiV4xo', 'Gn2W3X_pGh4', 'NCJTV5KaJJc', 'Fivy99RtMfM', 'LFJ9WAHowcg', 'fbAj9JfCXng', 'IWjzT2l5C34', 'd6K-sePuFAo', '5CSDIcUsIJk', 'LZXUR4z2P9w', 'MgnnQ2CN6yY', 'FV-c2FnPnDE', 'h27g5iT0tck', 'Gn2W3X_pGh4', 'N8Votwxx8a0', 'MgOVOCUuScE', 'fm_0sTNcDIo', 'h9SKyrHRhDo', '-Z-ul0GzzM4']

    #time_list = ['0:08:45.677978-0:08:53.844173', '0:13:45.333333-0:13:50.333333', '0:03:16.125000-0:03:21.458333', '0:04:31.958333-0:04:40.208333', '0:00:22.208333-0:00:30.416667', '0:03:22.833333-0:03:32.208333', '0:03:16.125000-0:03:21.458333', '0:04:31.958333-0:04:40.208333', '0:00:22.208333-0:00:30.416667', '0:17:45.166667-0:17:56.791667', '0:10:18.876255-0:10:27.009276', '0:06:55.625000-0:07:02.291667', '0:00:52.360000-0:01:00', '0:02:51.400000-0:03:03.200000', '0:02:41.916667-0:02:51.583333', '0:02:43.500000-0:02:55', '0:01:35.440000-0:01:44.120000', '0:04:35.708333-0:04:43.500000', '0:06:28.958333-0:06:40.875000', '0:05:18.820538-0:05:29.859864', '0:06:19.633333-0:06:30.333333', '0:01:08.675853-0:01:16.075406', '0:03:50.708333-0:03:57.708333', '0:04:38.343536-0:04:48.592377', '0:04:08.800000-0:04:20.120000', '0:09:47.933910-0:09:55.141944', '0:03:25.708333-0:03:32.916667', '0:13:51.840000-0:14:02.120000', '0:05:25.916667-0:05:35.458333', '0:05:58.958333-0:06:08.833333', '0:04:21.375000-0:04:32.625000', '0:10:49.300000-0:10:55.866667', '0:04:39.695478-0:04:49.778348', '0:09:01.250000-0:09:08.416667', '0:03:50.708333-0:03:57.708333', '0:11:37.333333-0:11:46.291667', '0:00:40.366974-0:00:46.074173', '0:05:35.733583-0:05:42.066606', '0:16:19.600000-0:16:27.333333', '0:10:13.056902-0:10:20.723239']

    #candi_list = [18, 406, 190, 455, 389, 299, 190, 455, 389, 554, 12, 48, 435, 319, 320, 360, 227, 
    #        271, 486, 132, 21, 32, 40, 49, 110, 131, 140, 154, 236, 662, 638, 16, 29, 30, 40, 70, 115, 129, 142, 188] 

    count_bar = tqdm(total=len(train_data_loader))
    word_gather = [] 
    gather_dict = {} 
    sub_id = 0
    for iter_idx, data in enumerate(train_data_loader):
        count_bar.update(1)
        time_seq, in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
        pdb.set_trace()
        #word_gather.extend(aux_info['word_list']) 
        #for cur_id in range(len(aux_info['vid'])):
        #    gather_dict[sub_id] = {} 
        #    gather_dict[sub_id]['vid'] = aux_info['vid'][cur_id]
        #    gather_dict[sub_id]['start_frame_no'] = aux_info['start_frame_no'][cur_id].item() 
        #    gather_dict[sub_id]['end_frame_no'] = aux_info['end_frame_no'][cur_id].item() 
        #    gather_dict[sub_id]['start_time'] = aux_info['start_time'][cur_id].item() 
        #    gather_dict[sub_id]['end_time'] = aux_info['end_time'][cur_id].item() 
        #    gather_dict[sub_id]['word_list'] = aux_info['word_list'][cur_id]
        #    sub_id += 1
    count_bar.close()


    count_bar = tqdm(total=len(val_data_loader))
    word_gather = [] 
    gather_dict = {} 
    sub_id = 0
    for iter_idx, data in enumerate(val_data_loader):
        count_bar.update(1)
        time_seq, in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
    count_bar.close()

    count_bar = tqdm(total=len(test_data_loader))
    word_gather = [] 
    gather_dict = {} 
    sub_id = 0
    for iter_idx, data in enumerate(test_data_loader):
        count_bar.update(1)
        time_seq, in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
    count_bar.close()








    #count_bar = tqdm(total=len(val_data_loader))
    #word_gather = [] 
    #gather_dict = {} 
    #sub_id = 0
    #for iter_idx, data in enumerate(val_data_loader):
    #    count_bar.update(1)
    #    time_seq, in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
    #    word_gather.extend(aux_info['word_list']) 
    #    gather_dict[sub_id] = {} 
    #    for cur_id in range(len(aux_info['vid'])):
    #        gather_dict[sub_id]['vid'] = aux_info['vid'][cur_id]
    #        gather_dict[sub_id]['start_frame_no'] = aux_info['start_frame_no'][cur_id].item() 
    #        gather_dict[sub_id]['end_frame_no'] = aux_info['end_frame_no'][cur_id].item() 
    #        gather_dict[sub_id]['start_time'] = aux_info['start_time'][cur_id].item() 
    #        gather_dict[sub_id]['end_time'] = aux_info['end_time'][cur_id].item() 
    #        gather_dict[sub_id]['word_list'] = aux_info['word_list'][cur_id]
    #        sub_id += 1
    #count_bar.close()
    #torch.save(word_gather, 'val_words.pt') 
    #torch.save(gather_dict, 'val_gather_dict.pt') 


    #count_bar = tqdm(total=len(test_data_loader))
    #word_gather = [] 
    #gather_dict = {} 
    #sub_id = 0
    #for iter_idx, data in enumerate(test_data_loader):
    #    count_bar.update(1)
    #    time_seq, in_text, text_lengths, in_text_padded, _, target_vec, in_audio, in_spec, aux_info = data
    #    word_gather.extend(aux_info['word_list']) 
    #    gather_dict[sub_id] = {} 
    #    for cur_id in range(len(aux_info['vid'])):
    #        gather_dict[sub_id]['vid'] = aux_info['vid'][cur_id]
    #        gather_dict[sub_id]['start_frame_no'] = aux_info['start_frame_no'][cur_id].item() 
    #        gather_dict[sub_id]['end_frame_no'] = aux_info['end_frame_no'][cur_id].item() 
    #        gather_dict[sub_id]['start_time'] = aux_info['start_time'][cur_id].item() 
    #        gather_dict[sub_id]['end_time'] = aux_info['end_time'][cur_id].item() 
    #        gather_dict[sub_id]['word_list'] = aux_info['word_list'][cur_id]
    #    sub_id += 1
    #count_bar.close()
    #torch.save(word_gather, 'test_words.pt') 
    #torch.save(gather_dict, 'test_gather_dict.pt') 




def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    # dataset config
    #if args.model == 'seq2seq':
        #collate_fn = word_seq_collate_fn
    #else:
        #collate_fn = default_collate_fn
    collate_fn = word_seq_add_beat_collate_fn

    # dataset
    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    train_dataset = SpeechMotionDataset(args.train_data_path[0],
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=args.mean_pose,
                                        remove_word_timing=(args.input_context == 'text'), 
                                        save_flag=False
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    val_dataset = SpeechMotionDataset(args.val_data_path[0],
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=train_dataset.speaker_model,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=args.mean_pose,
                                      remove_word_timing=(args.input_context == 'text'), 
                                      save_flag=False 
                                      )
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=False, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=collate_fn
                             )

    test_dataset = SpeechMotionDataset(args.test_data_path[0],
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       speaker_model=train_dataset.speaker_model,
                                       mean_dir_vec=mean_dir_vec,
                                       mean_pose=args.mean_pose, 
                                       save_flag=False 
                                       )

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=False, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=collate_fn
                             )
    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    pose_dim = 27  # 9 x 3
    train_epochs(args, train_loader, val_loader, test_loader, lang_model, mean_dir_vec, 
                 pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
