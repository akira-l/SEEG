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
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

from tqdm import tqdm 

import utils
from data_loader.lmdb_data_loader import SpeechMotionDataset, default_collate_fn, word_seq_collate_fn
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from train import evaluate_testset
from utils.data_utils import extract_melspectrogram, remove_tags_marks, convert_dir_vec_to_pose
from utils.train_utils import create_video_and_save, set_logger, create_video_and_save_single
from utils.tts_helper import TTSHelper

#sys.path.insert(0, '../../gentle')
#import gentle

from data_loader.data_preprocessor import DataPreprocessor

import pdb 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#gentle_resources = gentle.Resources()


def generate_gestures(args, pose_decoder, lang_model, audio, words, audio_sr=16000, vid=None,
                      seed_seq=None, fade_out=False):
    out_list = []
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr

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
    result_save_path = 'output/generation_results'

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

    if mode == 'eval':
        val_data_path = 'data/ted_dataset/lmdb_val'
        eval_net_path = 'output/train_h36m_gesture_autoencoder/gesture_autoencoder_checkpoint_best.bin'
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, eval_net_path, lang_model, device)
        val_dataset = load_dataset(val_data_path)
        data_loader = DataLoader(dataset=val_dataset, batch_size=32, collate_fn=collate_fn,
                                 shuffle=False, drop_last=True, num_workers=args.loader_workers)
        val_dataset.set_lang_model(lang_model)
        evaluate_testset(data_loader, generator, loss_fn, embed_space_evaluator, args)

    elif mode == 'from_text':
        random.seed()

        examples = [
            '<break time="0.5s"/><prosody>once handed me a very thick book. <break time="0.1s"/>it was his familys legacy</prosody>',
            '<break time="0.5s"/>we can help millions of teens with counseling',
            'what an amazing day that will be. what a big opportunity we have.',
            'just the way a surgeon operates on a patient you can literally interact with your table',
            '[Enter a new text]'
        ]

        if option:
            voice = option
        else:
            voice = 'en-female'

        vid = random.sample(range(0, speaker_model.n_words), 1)[0]
        tts = TTSHelper(cache_path='output/cached_wav')

        # text input
        for i, example in enumerate(examples):
            print('(%d) %s' % (i, example))

        try:
            select = int(input("select: "))
        except ValueError:
            exit(0)

        if select == len(examples) - 1:
            input_text = input("text: ")
        elif select >= len(examples) or select < 0:
            print('Please input a valid number. Exiting...')
            exit(0)
        else:
            input_text = examples[select]

        # generation
        text_without_tags = remove_tags_marks(input_text)
        #print(text_without_tags)

        tts_filename = tts.synthesis(input_text, voice_name=voice, verbose=False)
        sound_obj, duration = tts.get_sound_obj(tts_filename)
        print('TTS complete (audio length: {0:.1f}s)'.format(duration))
        audio, audio_sr = librosa.load(tts_filename, mono=True, sr=16000, res_type='kaiser_fast')

        words_with_timestamps = align_words(audio, text_without_tags)

        dir_vec = generate_gestures(args, generator, lang_model, audio, words_with_timestamps, vid=vid,
                                    fade_out=False)

        # make a video
        save_path = 'output/generation_results'
        os.makedirs(save_path, exist_ok=True)
        prefix = '{}_vid_{}_{}'.format(text_without_tags[:50], vid, voice)
        out_pos, _ = create_video_and_save(
            save_path, 0, prefix, 0, None, dir_vec, mean_dir_vec, text_without_tags, audio=audio,
            clipping_to_shortest_stream=True, delete_audio_file=False)

        # save pkl
        save_dict = {
            'sentence': words_with_timestamps, 'audio': audio,
            'out_dir_vec': dir_vec + mean_dir_vec, 'out_poses': out_pos,
            'aux_info': ''
        }
        with open(os.path.join(result_save_path, '{}.pkl'.format(prefix)), 'wb') as f:
            pickle.dump(save_dict, f)

    elif mode == 'from_db_clip':

        vid_list = ['d38LKbYfWrs', 'awADEuv5vWY', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'B905LapVP7I', 'jAemh_JxgOk', 'rSQNi5sAwuc', 'zHbkOWz6AAg', 'gVfgkFaswn4', 'LujWrkYsl64', 'GMynksvCcUI', '1mLQFm3wEfw', 'fWqKalpYgLo', 'YATYsgi3e5A', '7MHOk7qVhYs', 'uTbA-mxo858', 'HR9956gDpUY', 'YUUP2MMz7PU', '8Z24LCysq3A', '1N39Z0ODeME', '1L6l-FiV4xo', 'Gn2W3X_pGh4', 'NCJTV5KaJJc', 'Fivy99RtMfM', 'LFJ9WAHowcg', 'fbAj9JfCXng', 'IWjzT2l5C34', 'd6K-sePuFAo', '5CSDIcUsIJk', 'LZXUR4z2P9w', 'MgnnQ2CN6yY', 'FV-c2FnPnDE', 'h27g5iT0tck', 'Gn2W3X_pGh4', 'N8Votwxx8a0', 'MgOVOCUuScE', 'fm_0sTNcDIo', 'h9SKyrHRhDo', '-Z-ul0GzzM4']

        time_list = ['0:08:45.677978-0:08:53.844173', '0:13:45.333333-0:13:50.333333', '0:03:16.125000-0:03:21.458333', '0:04:31.958333-0:04:40.208333', '0:00:22.208333-0:00:30.416667', '0:03:22.833333-0:03:32.208333', '0:03:16.125000-0:03:21.458333', '0:04:31.958333-0:04:40.208333', '0:00:22.208333-0:00:30.416667', '0:17:45.166667-0:17:56.791667', '0:10:18.876255-0:10:27.009276', '0:06:55.625000-0:07:02.291667', '0:00:52.360000-0:01:00', '0:02:51.400000-0:03:03.200000', '0:02:41.916667-0:02:51.583333', '0:02:43.500000-0:02:55', '0:01:35.440000-0:01:44.120000', '0:04:35.708333-0:04:43.500000', '0:06:28.958333-0:06:40.875000', '0:05:18.820538-0:05:29.859864', '0:06:19.633333-0:06:30.333333', '0:01:08.675853-0:01:16.075406', '0:03:50.708333-0:03:57.708333', '0:04:38.343536-0:04:48.592377', '0:04:08.800000-0:04:20.120000', '0:09:47.933910-0:09:55.141944', '0:03:25.708333-0:03:32.916667', '0:13:51.840000-0:14:02.120000', '0:05:25.916667-0:05:35.458333', '0:05:58.958333-0:06:08.833333', '0:04:21.375000-0:04:32.625000', '0:10:49.300000-0:10:55.866667', '0:04:39.695478-0:04:49.778348', '0:09:01.250000-0:09:08.416667', '0:03:50.708333-0:03:57.708333', '0:11:37.333333-0:11:46.291667', '0:00:40.366974-0:00:46.074173', '0:05:35.733583-0:05:42.066606', '0:16:19.600000-0:16:27.333333', '0:10:13.056902-0:10:20.723239']

        candi_list = [18, 406, 190, 455, 389, 299, 190, 455, 389, 554, 12, 48, 435, 319, 320, 360, 227, 
                271, 486, 132, 21, 32, 40, 49, 110, 131, 140, 154, 236, 662, 638, 16, 29, 30, 40, 70, 115, 129, 142, 188] 

        save_mode_list = ['train', 'val', 'test']  
        save_mode = save_mode_list[data_version]
        test_data_path = 'data/ted_dataset/lmdb_' + save_mode 
        #test_data_path = 'data/ted_dataset/lmdb_test'
        #test_data_path = 'data/ted_dataset/lmdb_train'
        #test_data_path = 'data/ted_dataset/lmdb_val'
        save_path = 'output/generation_results'
        clip_duration_range = [5, 12]
        random.seed()

        if option:
            n_generations = int(option)
        else:
            n_generations = 5

        # load clips and make gestures
        n_saved = 0
        save_counter = 0 
        lmdb_env = lmdb.open(test_data_path, readonly=True, lock=False)
        handle = open(save_mode+ '_save_full.txt', 'a')
        sub_folder_save_name = save_mode + '_full' 
        if not os.path.exists(sub_folder_save_name): 
            os.makedirs(sub_folder_save_name) 

        with lmdb_env.begin(write=False) as txn:
            keys = [key for key, _ in txn.cursor()]
            #while n_saved < n_generations:  # loop until we get the desired number of results
            count_bar = tqdm(total=len(keys))
            for key in keys: 
                count_bar.update(1) 
                # select video
                #key = random.choice(keys)

                buf = txn.get(key)
                video = pyarrow.deserialize(buf)
                vid = video['vid']
                clips = video['clips']

                # select clip
                n_clips = len(clips)
                if n_clips == 0:
                    continue
                #clip_idx = random.randrange(n_clips)

                for clip_idx in range(n_clips):

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

                    vid_idx = random.sample(range(0, speaker_model.n_words), 1)[0]
                    out_dir_vec = generate_gestures(args, generator, lang_model, clip_audio, clip_words, vid=vid_idx,
                                                    seed_seq=target_dir_vec[0:args.n_pre_poses], fade_out=False)

                    # make a video
                    sentence_words = []
                    for word, _, _ in clip_words:
                        sentence_words.append(word)
                    sentence = ' '.join(sentence_words)

                    os.makedirs(save_path, exist_ok=True)

                    filename_prefix = '{}_{}_{}'.format(vid, vid_idx, clip_idx)
                    filename_prefix_for_video = filename_prefix
                    aux_str = '({}, time: {}-{})'.format(vid, str(datetime.timedelta(seconds=clip_time[0])),
                                                         str(datetime.timedelta(seconds=clip_time[1])))

                    st_en_time_key = '{}-{}'.format(str(datetime.timedelta(seconds=clip_time[0])),
                                                    str(datetime.timedelta(seconds=clip_time[1])) ) 

                    #create_video_and_save(
                    #    save_path, 0, filename_prefix_for_video, 0, target_dir_vec, out_dir_vec,
                    #    mean_dir_vec, sentence, audio=clip_audio, aux_str=aux_str,
                    #    clipping_to_shortest_stream=True, delete_audio_file=False)

                    save_folder_name = save_mode + '_save' 
                    if not os.path.exists(save_folder_name): 
                        os.makedirs(save_folder_name) 

                    #create_video_and_save_single(
                    #    save_folder_name, str(save_counter), '',
                    #    target_dir_vec,
                    #    mean_dir_vec, sentence, audio=clip_audio, aux_str=aux_str,
                    #    clipping_to_shortest_stream=True, delete_audio_file=False)
                    write_cont = str(save_counter) + '  ' + sentence + '  ' + aux_str + '\n' 
                    handle.write(write_cont) 


                    #if st_en_time_key in time_list and vid in vid_list:  
                    #    sub_candi_ind = time_list.index(st_en_time_key) 
                    #    candi_ind_ = candi_list[sub_candi_ind]  

                    #    overall_gesture_inds = target_dir_vec.shape[0] 
                    #    overall_audio_inds = clip_audio.shape[0]  
                    #    for sub_word_meta in clip_words: 
                    #        sub_word_text = sub_word_meta[0]
                    #        sub_word_time = sub_word_meta[2] - sub_word_meta[1] 
                    #        stay_ratio = sub_word_time / clip_duration 

                    #        sub_word_meta[1] = max(0, sub_word_meta[1]) 
                    #        start_ratio = sub_word_meta[1] / clip_duration 
                    #        start_ind = int(overall_gesture_inds * start_ratio) 

                    #        stay_ind = int(overall_gesture_inds * stay_ratio) 
                    #        stay_ind = max(1, start_ind)  

                    #        stay_audio_ind = int(overall_audio_inds * stay_ratio) 
                    #        stay_audio_ind = max(1, stay_audio_ind)  
                    #        start_audio_ind = int(overall_audio_inds * start_ratio) 

                    #        sub_duration_str = str(candi_ind_) + '_' + sub_word_text + '-' + str(start_ind)  
                    #        sub_folder_save_name = 'sub_' + str(candi_ind_) 
                    #        if not os.path.exists(sub_folder_save_name): 
                    #            os.makedirs(sub_folder_save_name) 

                    #        create_video_and_save_single(
                    #            sub_folder_save_name, sub_duration_str, '',
                    #            target_dir_vec[start_ind:(start_ind+stay_ind)],
                    #            mean_dir_vec, sub_word_text, 
                    #            audio=clip_audio[start_audio_ind:(start_audio_ind+stay_audio_ind)], 
                    #            aux_str=aux_str,
                    #            clipping_to_shortest_stream=True, delete_audio_file=False, save_flag=True)
 

                    create_video_and_save_single(
                            sub_folder_save_name, str(save_counter), '',
                            target_dir_vec,
                            mean_dir_vec, sentence, audio=clip_audio, aux_str=aux_str,
                            clipping_to_shortest_stream=True, delete_audio_file=False, save_flag=True)

                    save_counter += 1


                    # save pkl
                    out_dir_vec = out_dir_vec + mean_dir_vec
                    out_poses = convert_dir_vec_to_pose(out_dir_vec)

                    save_dict = {
                        'sentence': sentence, 'audio': clip_audio.astype(np.float32),
                        'out_dir_vec': out_dir_vec, 'out_poses': out_poses,
                        'aux_info': '{}_{}_{}'.format(vid, vid_idx, clip_idx),
                        'human_dir_vec': target_dir_vec + mean_dir_vec,
                    }
                    #with open(os.path.join(save_path, '{}.pkl'.format(filename_prefix)), 'wb') as f:
                        #pickle.dump(save_dict, f)

                    n_saved += 1
        count_bar.close() 
        handle.close() 
    else:
        assert False, 'wrong mode'


if __name__ == '__main__':
    mode = sys.argv[1]  # {eval, from_db_clip, from_text}
    ckpt_path = sys.argv[2]

    option = None
    if len(sys.argv) > 3:
        option = sys.argv[3]

    set_logger()
    for data_id in range(3): 
        main(mode, ckpt_path, option, data_id)
