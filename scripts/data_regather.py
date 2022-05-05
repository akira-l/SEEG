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
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator

from torch import optim

from data_loader.lmdb_data_loader import *
import utils.train_utils

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


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]
    best_val_loss = (1e+10, 0)  # value, epoch

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 20

    # z type
    if args.z_type == 'speaker':
        pass
    elif args.z_type == 'random':
        speaker_model = 1
    else:
        speaker_model = None

    # init model
    generator, discriminator, loss_fn = init_model(args, lang_model, speaker_model, pose_dim, device)

    # use multi GPUs
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator)

    # prepare an evaluator for FGD
    embed_space_evaluator = None
    if args.eval_net_path and len(args.eval_net_path) > 0:
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    
    evaluate_sample_and_save_video(
        0, args.name, test_data_loader, generator,
        args=args, lang_model=lang_model)

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))



def evaluate_sample_and_save_video(epoch, prefix, test_data_loader, generator, args, lang_model,
                                   n_save=None, save_path=None):
    generator.train(False)  # eval mode
    start = time.time()
    if not n_save:
        n_save = 1 if epoch <= 0 else 5

    out_raw = []
    save_map_dict = {} 
    
    err_path = 'save_err'
    if not os.path.exists(err_path): 
        os.makedirs(err_path)
    
    count_bar = tqdm(total=len(test_data_loader))
    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader):

            words_list, all_in_text, all_text_lengths, all_in_text_padded, _, all_target_dir_vec, all_in_audio, all_in_spec, aux_info = data

            # prepare
            count_bar.update(1) 
            
            count_bar_sub = tqdm(total=len(words_list))
            for select_index in range(len(words_list)):
                
                sub_file_name = str(iter_idx) + '_' + str(select_index)
                count_bar_sub.update(1)

                if args.model == 'seq2seq':
                    in_text = all_in_text[select_index, :].unsqueeze(0).to(device)
                    text_lengths = all_text_lengths[select_index].unsqueeze(0).to(device)
                in_text_padded = all_in_text_padded[select_index, :].unsqueeze(0).to(device)
                in_audio = all_in_audio[select_index, :].unsqueeze(0).to(device)
                in_spec = all_in_spec[select_index, :, :].unsqueeze(0).to(device)
                target_dir_vec = all_target_dir_vec[select_index, :, :].unsqueeze(0).to(device)

                input_words = []
                for i in range(all_in_text_padded.shape[1]):
                    word_idx = int(all_in_text_padded.data[select_index, i])
                    if word_idx > 0:
                        input_words.append(lang_model.index2word[word_idx])
                sentence = ' '.join(input_words)

                # speaker input
                speaker_model = utils.train_utils.get_speaker_model(generator)
                if speaker_model:
                    vid = aux_info['vid'][select_index]
                    # vid_indices = [speaker_model.word2index[vid]]
                    vid_indices = [random.choice(list(speaker_model.word2index.values()))]
                    vid_indices = torch.LongTensor(vid_indices).to(device)
                else:
                    vid_indices = None

                # aux info
                aux_str = '({}, time: {}-{})'.format(
                    aux_info['vid'][select_index],
                    str(datetime.timedelta(seconds=aux_info['start_time'][select_index].item())),
                    str(datetime.timedelta(seconds=aux_info['end_time'][select_index].item())))

                # synthesize
                pre_seq = target_dir_vec.new_zeros((target_dir_vec.shape[0], target_dir_vec.shape[1],
                                                    target_dir_vec.shape[2] + 1))
                pre_seq[:, 0:args.n_pre_poses, :-1] = target_dir_vec[:, 0:args.n_pre_poses]
                pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
                pre_seq_partial = pre_seq[:, 0:args.n_pre_poses, :-1]

                # to video
                audio_npy = np.squeeze(in_audio.cpu().numpy())
                target_dir_vec = np.squeeze(target_dir_vec.cpu().numpy())

                if save_path is None:
                    save_path = args.model_save_path
                
                save_file_name = '{}_{:03d}_{}'.format(prefix, epoch, iter_idx)
                save_map_dict[sub_file_name] = aux_str 
                
                mean_data = np.array(args.mean_dir_vec).reshape(-1, 3)
                try:
                    utils.train_utils.create_video_and_save(
                        save_path, epoch, prefix+'_'+sub_file_name, iter_idx,
                        target_dir_vec, target_dir_vec, mean_data,
                        sentence, audio=audio_npy, aux_str=aux_str)
                except: 
                    try: 
                        utils.train_utils.create_video_and_save(
                            save_path, epoch, prefix+'_'+sub_file_name, iter_idx,
                            target_dir_vec, target_dir_vec, mean_data,
                            sentence, audio=audio_npy, aux_str=aux_str)
                    except: 
                        meta_data_dict = {}
                        meta_data_dict['sent'] = sentence
                        meta_data_dict['aux'] = aux_str 
                        meta_data_dict['mean_data'] = mean_data
                        meta_data_dict['target_dir_vec'] = target_dir_vec
                        meta_data_dict['audio_npy'] = audio_npy 
                        torch.save(meta_data_dict, 
                                   os.path.join(err_path, str(iter_idx) + '.pt')
                                  )
                        print('meet error num: ', iter_idx, ' | aux_str: ', aux_str )

            count_bar_sub.close() 

    count_bar.close() 
    logging.info('saved sample videos, took {:.1f}s'.format(time.time() - start))
    
    torch.save(save_map_dict, 'save_name-map_dict.pt')

    return out_raw


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    # dataset config
    if args.model == 'seq2seq':
        collate_fn = word_seq_collate_fn
    else:
        collate_fn = default_collate_fn

    # dataset
    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    train_dataset = SpeechMotionDataset(args.train_data_path[0],
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=args.mean_pose,
                                        remove_word_timing=(args.input_context == 'text')
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    val_dataset = SpeechMotionDataset(args.val_data_path[0],
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=train_dataset.speaker_model,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=args.mean_pose,
                                      remove_word_timing=(args.input_context == 'text')
                                      )
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=collate_fn
                             )

    test_dataset = SpeechMotionDataset(args.test_data_path[0],
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       speaker_model=train_dataset.speaker_model,
                                       mean_dir_vec=mean_dir_vec,
                                       mean_pose=args.mean_pose)
    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    pose_dim = 27  # 9 x 3
    train_epochs(args, train_loader, test_loader, lang_model,
                 pose_dim=pose_dim, speaker_model=train_dataset.speaker_model)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
