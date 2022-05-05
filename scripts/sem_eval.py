import datetime
import logging
import math
import os
import pickle
import random
import sys

import librosa
import soundfile as sf
import lmdb
import numpy as np
import time

import pyarrow
import torch
from torch.utils.data import DataLoader

import soundfile as sf
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

from tqdm import tqdm 

import utils
from data_loader.lmdb_data_loader import SpeechMotionDataset, default_collate_fn, word_seq_collate_fn
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
#from overall_train import evaluate_testset
from utils.data_utils import extract_melspectrogram, remove_tags_marks, convert_dir_vec_to_pose
from utils.train_utils import create_video_and_save, set_logger, create_video_and_save_single
from utils.tts_helper import TTSHelper

#sys.path.insert(0, '../../gentle')
#import gentle

from data_loader.data_preprocessor import DataPreprocessor

import pdb 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#gentle_resources = gentle.Resources()

gen_test_cont_handle = open('scripts/ges_test.txt') 
gen_test_cont = gen_test_cont_handle.readlines() 
gen_test_cont = [x[:-1] for x in gen_test_cont] 
gen_test_vid_name = [x[:11] for x in gen_test_cont] 
gen_test_vid_idx = [x[12:].split('_')[0] for x in gen_test_cont] 
gen_test_clip_idx = [x[12:].split('_')[1] for x in gen_test_cont] 

gen_test_name_idx_dict = {} 
gen_test_name_clip_dict = {} 
for sub_name, sub_idx, sub_clip_idx in zip(gen_test_vid_name, gen_test_vid_idx, gen_test_clip_idx): 
    if sub_name not in gen_test_name_idx_dict: 
        gen_test_name_idx_dict[sub_name] = [int(sub_idx)] 
        gen_test_name_clip_dict[sub_name] = [int(sub_clip_idx)] 
    else: 
        gen_test_name_idx_dict[sub_name].append(int(sub_idx))
        gen_test_name_clip_dict[sub_name].append(int(sub_clip_idx)) 


get_sim = nn.CosineSimilarity(dim=1) 

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")




def generate_gestures(args, pose_decoder, lang_model, aux_info, audio, words, audio_sr=16000, vid=None,
                      seed_seq=None, fade_out=False):
    out_list = []
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr

    audio_len = 340*3
    ind = torch.arange(audio_len) 
    word_list = [x[0] for x in words] 
 
    use_spectrogram = False
    if args.model == 'speech2gesture':
        use_spectrogram = True

    # pre seq
    pre_seq = torch.zeros((1, n_frames, len(args.mean_dir_vec) + 1))
    if seed_seq is not None:
        pre_seq[0, 0:args.n_pre_poses, :-1] = torch.Tensor(seed_seq[0:args.n_pre_poses])
        pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for seed poses

    sr = 16000
    spectrogram = None
    if use_spectrogram:
        # audio to spectrogram
        spectrogram = extract_melspectrogram(audio, sr)

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    spectrogram_sample_length = int(round(unit_time * sr / 512))
    audio_sample_length = int(unit_time * audio_sr)
    end_padding_duration = 0

    # prepare speaker input
    if args.z_type == 'speaker':
        if not vid:
            vid = random.randrange(pose_decoder.z_obj.n_words)
        #print('vid:', vid)
        vid = torch.LongTensor([vid]).to(device)
    else:
        vid = None

    #print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

    out_dir_vec = None
    start = time.time()
    for i in range(0, num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time

        # prepare spectrogram input
        in_spec = None
        if use_spectrogram:
            # prepare spec input
            audio_start = math.floor(start_time / clip_length * spectrogram.shape[0])
            audio_end = audio_start + spectrogram_sample_length
            in_spec = spectrogram[:, audio_start:audio_end]
            in_spec = torch.from_numpy(in_spec).unsqueeze(0).to(device)

        # prepare audio input
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        if len(in_audio) < audio_sample_length:
            if i == num_subdivision - 1:
                end_padding_duration = audio_sample_length - len(in_audio)
            in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
        audio_numpy = in_audio.copy().astype(np.float32) 
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()


        audio_flag = (in_audio > 1e-3).float() 
        audio_thresh = (in_audio * audio_flag).sum() / audio_flag.sum() 

        audio_len = 340*3
        ind = torch.arange(audio_len) 
        au_size = in_audio.size(1)
        audio_step = au_size // audio_len  
        ind_step = ind * audio_step 
        sample_bias = audio_step // 2
        ind_step_bias = ind_step + sample_bias  
        ind_step_bias[-1] = au_size - 1 if ind_step_bias[-1] >= au_size else ind_step_bias[-1]
        audio_sample = in_audio[0].abs()[ind_step_bias]
        audio_flag = (audio_sample > audio_thresh).float() 

        
        hop_len = au_size // audio_len  
        sr_num = 16000
        oenv = librosa.onset.onset_strength(y=audio_numpy, sr=sr_num, hop_length=hop_len) 
        start_skip = (oenv.size - audio_len) // 2
        audio_oenv = torch.from_numpy(oenv)[start_skip:audio_len+start_skip] 

        audio_beat = torch.stack([audio_flag, audio_oenv.cuda() ])  

        time_seq = audio_beat.unsqueeze(0)


        # prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        frame_duration = (end_time - start_time) / n_frames
        for w_i, word in enumerate(word_seq):
            #print(word[0], end=', ')
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            extended_word_indices[idx] = lang_model.get_word_index(word[0])
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        #print(' ')
        in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(device)
        in_text = torch.LongTensor(word_indices).unsqueeze(0).to(device)

        # prepare pre seq
        if i > 0:
            pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
            pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq = pre_seq.float().to(device)
        pre_seq_partial = pre_seq[0, 0:args.n_pre_poses, :-1].unsqueeze(0)

        # synthesize
        #print(in_text_padded)
        if args.model == 'multimodal_context':
            out_dir_vec, *_ = pose_decoder(pre_seq, time_seq, in_text_padded, in_audio, vid)
        elif args.model == 'joint_embedding':
            _, _, _, _, _, _, out_dir_vec = pose_decoder(in_text_padded, in_audio, pre_seq_partial, None, 'speech')
        elif args.model == 'seq2seq':
            words_lengths = torch.LongTensor([in_text.shape[1]]).to(device)
            out_dir_vec = pose_decoder(in_text, words_lengths, pre_seq_partial, None)
        elif args.model == 'speech2gesture':
            out_dir_vec = pose_decoder(in_spec, pre_seq_partial)
        else:
            assert False

        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)

    #print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    out_dir_vec = np.vstack(out_list)

    # additional interpolation for seq2seq
    if args.model == 'seq2seq':
        n_smooth = args.n_pre_poses
        for i in range(num_subdivision):
            start_frame = args.n_pre_poses + i * (args.n_poses - args.n_pre_poses) - n_smooth
            if start_frame < 0:
                start_frame = 0
                end_frame = start_frame + n_smooth * 2
            else:
                end_frame = start_frame + n_smooth * 3

            # spline interp
            y = out_dir_vec[start_frame:end_frame]
            x = np.array(range(0, y.shape[0]))
            w = np.ones(len(y))
            w[0] = 5
            w[-1] = 5

            coeffs = np.polyfit(x, y, 3)
            fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
            interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
            interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

            out_dir_vec[start_frame:end_frame] = interpolated_y

    # fade out to the mean pose
    if fade_out:
        n_smooth = args.n_pre_poses
        start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
        end_frame = start_frame + n_smooth * 2
        if len(out_dir_vec) < end_frame:
            out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode='constant')
        out_dir_vec[end_frame-n_smooth:] = np.zeros((len(args.mean_dir_vec)))  # fade out to mean poses

        # interpolation
        y = out_dir_vec[start_frame:end_frame]
        x = np.array(range(0, y.shape[0]))
        w = np.ones(len(y))
        w[0] = 5
        w[-1] = 5
        coeffs = np.polyfit(x, y, 2, w=w)
        fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
        interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
        interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

        out_dir_vec[start_frame:end_frame] = interpolated_y

    return out_dir_vec







def align_words(audio, text):
    # resample audio to 8K
    audio_8k = librosa.resample(audio, 16000, 8000)
    wave_file = 'output/temp.wav'
    sf.write(wave_file, audio_8k, 8000, 'PCM_16')

    # run gentle to align words
    aligner = gentle.ForcedAligner(gentle_resources, text, nthreads=2, disfluency=False,
                                   conservative=False)
    gentle_out = aligner.transcribe(wave_file, logging=logging)
    words_with_timestamps = []
    for i, gentle_word in enumerate(gentle_out.words):
        if gentle_word.case == 'success':
            words_with_timestamps.append([gentle_word.word, gentle_word.start, gentle_word.end])
        elif 0 < i < len(gentle_out.words) - 1:
            words_with_timestamps.append([gentle_word.word, gentle_out.words[i-1].end, gentle_out.words[i+1].start])

    return words_with_timestamps




def main(mode, checkpoint_path, option, data_version=2):
    args, generator, loss_fn, lang_model, speaker_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path, device)
    result_save_path = 'output/generation_results_comb'

    # load mean vec
    mean_pose = np.array(args.mean_pose).squeeze()
    mean_dir_vec = np.array(args.mean_dir_vec).squeeze()

    # load lang_model
    vocab_cache_path = os.path.join('data/ted_dataset', 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    if args.model == 'seq2seq':
        collate_fn = word_seq_collate_fn
    else:
        collate_fn = default_collate_fn

    def load_dataset(path):
        dataset = SpeechMotionDataset(path,
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=speaker_model,
                                      mean_pose=mean_pose,
                                      mean_dir_vec=mean_dir_vec
                                      )
        print(len(dataset))
        return dataset


    if mode == 'from_db_clip':

        save_mode_list = ['train', 'val', 'test']  
        save_mode = save_mode_list[data_version]
        test_data_path = 'data/ted_dataset/lmdb_' + save_mode 
        #test_data_path = 'data/ted_dataset/lmdb_test'
        #test_data_path = 'data/ted_dataset/lmdb_train'
        #test_data_path = 'data/ted_dataset/lmdb_val'
        #save_path = 'output/re_generation_results_' + dt() 
        clip_duration_range = [5, 12]
        random.seed()

        if option:
            n_generations = int(option)
        else:
            n_generations = 5

        # load clips and make gestures
        lmdb_env = lmdb.open(test_data_path, readonly=True, lock=False)
        #handle = open(save_mode+ '_save.txt', 'a')

        gather_test_pose = [] 

        with lmdb_env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]
            count_bar = tqdm(total=len(keys))
            for key in keys: 
                count_bar.update(1) 
                # select video
                #key = random.choice(keys)

                buf = txn.get(key)
                video = pyarrow.deserialize(buf)
                vid = video['vid']
                clips = video['clips']

                if vid not in gen_test_name_idx_dict: 
                    continue 

                # select clip
                n_clips = len(clips)
                if n_clips == 0:
                    continue
                #clip_idx = random.randrange(n_clips)

                vid_idx = 0 
                for clip_idx in range(n_clips):
                #for vid_idx, clip_idx in zip(gen_test_name_idx_dict[vid], gen_test_name_clip_dict[vid]): 

                    clip_poses = clips[clip_idx]['skeletons_3d']
                    clip_audio = clips[clip_idx]['audio_raw']
                    clip_words = clips[clip_idx]['words']
                    clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]

                    clip_poses = utils.data_utils.resample_pose_seq(clip_poses, clip_time[1] - clip_time[0],
                                                                    args.motion_resampling_framerate)
                    target_dir_vec = utils.data_utils.convert_pose_seq_to_dir_vec(clip_poses)
                    target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
                    target_dir_vec -= mean_dir_vec

                    # check duration
                    clip_duration = clip_time[1] - clip_time[0]
                    if clip_duration < clip_duration_range[0] or clip_duration > clip_duration_range[1]:
                        continue

                    # synthesize
                    for selected_vi in range(len(clip_words)):  # make start time of input text zero
                        clip_words[selected_vi][1] -= clip_time[0]  # start time
                        clip_words[selected_vi][2] -= clip_time[0]  # end time


                    clips[clip_idx]['vid'] = vid 

                    #out_dir_vec = generate_gestures(args, generator, lang_model, clip_audio, clip_words, vid=vid_idx,
                    #                                seed_seq=target_dir_vec[0:args.n_pre_poses], fade_out=False)

                    out_dir_vec = generate_gestures(args, generator, lang_model, clips[clip_idx], clip_audio, clip_words, vid=vid_idx,
                                    fade_out=False)

                    gather_test_pose.append(torch.from_numpy(out_dir_vec).mean(0))

        count_bar.close() 
        #handle.close() 

    else:
        assert False, 'wrong mode'


if __name__ == '__main__':
    mode = sys.argv[1]  # {eval, from_db_clip, from_text}
    ckpt_path = sys.argv[2]

    option = None
    if len(sys.argv) > 3:
        option = sys.argv[3]

    set_logger()
    main(mode, ckpt_path, option, 2)
    #for data_id in range(3): 
    #    main(mode, ckpt_path, option, data_id)
    #    break 
