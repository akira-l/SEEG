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

import utils.train_utils
import utils.data_utils
from model.vocab import Vocab
from data_loader.data_preprocessor import DataPreprocessor
import pyarrow

import librosa

import pdb 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def word_seq_add_beat_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    beats, word_seq, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    beats = default_collate(beats) 
    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return beats, word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info





def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    _, word_seq, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info


def default_collate_fn(data):
    _, text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info


class SpeechMotionDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec,
                 speaker_model=None, remove_word_timing=False, save_flag=False):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_dir_vec = mean_dir_vec
        self.remove_word_timing = remove_word_timing

        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = utils.data_utils.calc_spectrogram_length_from_motion_length(
            n_poses, pose_resampling_fps)

        self.lang_model = None
        self.save_flag = save_flag  

        #self.beat_path = 'beat_resave' 
        self.beat_path = '../Gesture-Generation-from-Trimodal-Context/double_feat' 

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            logging.info('Creating the dataset cache...')
            assert mean_dir_vec is not None
            if mean_dir_vec.shape[-1] != 3:
                mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))
            n_poses_extended = int(round(n_poses * 1.25))  # some margin
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses_extended,
                                            subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec)
            data_sampler.run()
        else:
            logging.info('Found the cache {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # make a speaker model
        if speaker_model is None or speaker_model == 0:
            precomputed_model = lmdb_dir + '_speaker_model.pkl'
            if not os.path.exists(precomputed_model):
                self._make_speaker_model(lmdb_dir, precomputed_model)
            else:
                with open(precomputed_model, 'rb') as f:
                    self.speaker_model = pickle.load(f)
        else:
            self.speaker_model = speaker_model

    def __len__(self):
        return self.n_samples


    def proc_audio(self, audio, audio_numpy): 
        audio_flag = (audio > 1e-3).float() 
        audio_thresh = (audio * audio_flag).sum() / audio_flag.sum() 

        audio_len = 340*3
        ind = torch.arange(audio_len) 
        au_size = audio.size(0)
        audio_step = au_size // audio_len  
        ind_step = ind * audio_step 
        sample_bias = audio_step // 2
        ind_step_bias = ind_step + sample_bias  
        ind_step_bias[-1] = au_size - 1 if ind_step_bias[-1] >= au_size else ind_step_bias[-1]
        audio_sample = audio.abs()[ind_step_bias]
        audio_flag = (audio_sample > audio_thresh).float() 

        
        hop_len = au_size // audio_len  
        sr_num = 16000
        oenv = librosa.onset.onset_strength(y=audio_numpy, sr=sr_num, hop_length=hop_len) 
        start_skip = (oenv.size - audio_len) // 2
        audio_oenv = torch.from_numpy(oenv)[start_skip:audio_len+start_skip] 

        audio_beat = torch.stack([audio_flag, audio_oenv])  

        return audio_beat




    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
            

        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            if self.remove_word_timing:
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        n_words += 1
                space = int(n_frames / (n_words + 1))
                for i in range(n_words):
                    idx = (i+1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[i][0])
            else:
                prev_idx = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                        prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        duration = aux_info['end_time'] - aux_info['start_time']
        do_clipping = True

        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
            audio = utils.data_utils.make_audio_fixed_length(audio, self.expected_audio_length)
            spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
            vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
        else:
            sample_end_time = None

        #word_list = [] 
        word_list = [x[0] for x in word_seq]
        start_time_list = [str(x[1]) for x in word_seq] 
        end_time_list = [str(x[2]) for x in word_seq] 
        sent = ' '.join(word_list)

        rec_sent = '|'.join(word_list) 
        rec_start = '|'.join(start_time_list)
        rec_end = '|'.join(end_time_list) 
        word_raw = rec_sent + ' ' + rec_start + ' ' + rec_end 

        aux_info['word_list'] = sent 
        aux_info['word_raw'] = word_raw  

        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)
        extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)
        vec_seq_ = torch.from_numpy(vec_seq.copy())
        vec_seq_ = vec_seq_.reshape((vec_seq_.shape[0], -1)).float()
        pose_seq = torch.from_numpy(pose_seq.copy()).reshape((pose_seq.shape[0], -1)).float()
        audio_numpy = audio.copy().astype(np.float32) 
        audio = torch.from_numpy(audio.copy()).float()

        aux_key = '_'.join([aux_info['vid'], 
                            str(aux_info['start_frame_no']), 
                            str(aux_info['end_frame_no'])])
        
        if os.path.exists(self.beat_path): 
            audio_beat = torch.load(os.path.join(self.beat_path, 
                                                 aux_key + '.pt')) 
        else: 
            audio_beat = self.proc_audio(audio, audio_numpy) 
            torch.save(audio_beat, os.path.join(self.beat_path, aux_key + '.pt')) 
            #sr_num = 16000
            #duration = librosa.get_duration(y=audio_numpy, sr=sr_num) 
            #try: 
            #    tempo, beats = librosa.beat.beat_track(y=audio_numpy, sr=sr_num)
            #    time_seq = librosa.frames_to_time(beats, sr=sr_num)
            #except: 
            #    time_seq = None

            #seq_len_ = 340 
            #time_seq_tensor = torch.zeros(seq_len_) 
            #if time_seq is not None: 
            #    time_seq_round = (seq_len_ * (time_seq / duration)).round() 
            #    for cnt_, ind in enumerate(time_seq_round): 
            #        ind = int(ind) 
            #        time_seq_tensor[ind:] = time_seq_tensor[ind:] + (-1)**cnt_ 
            #print(time_seq_round) 

        if self.save_flag: 
            torch.save(audio_beat, 
                       os.path.join(self.beat_path, aux_key + '.pt') ) 

        spectrogram = torch.from_numpy(spectrogram.copy())

        return audio_beat, word_seq_tensor, extended_word_seq, pose_seq, vec_seq_, audio, spectrogram, aux_info

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model

    def _make_speaker_model(self, lmdb_dir, cache_path):
        logging.info('  building a speaker model...')
        speaker_model = Vocab('vid', insert_default_tokens=False)

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        txn = lmdb_env.begin(write=False)
        cursor = txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            speaker_model.index_word(vid)

        lmdb_env.close()
        logging.info('    indexed %d videos' % speaker_model.n_words)
        self.speaker_model = speaker_model

        # cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.speaker_model, f)
