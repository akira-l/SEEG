import random

import numpy as np
import torch
import torch.nn.functional as F

import pdb 


def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise


def train_iter_gan(args, epoch, time_seq, in_text, in_audio, target_poses, vid_indices,
                   pose_decoder, discriminator,
                   pose_dec_optim, dis_optim):
    warm_up_epochs = args.loss_warmup
    use_noisy_target = False

    # make pre seq input
    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    ###########################################################################################
    # train D
    dis_error = None
    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        dis_optim.zero_grad()

        out_dir_vec, beat_out, *_ = pose_decoder(pre_seq, time_seq, in_text, in_audio, vid_indices)  # out shape (batch x seq x dim)

        if use_noisy_target:
            noise_target = add_noise(target_poses)
            noise_out = add_noise(out_dir_vec.detach())
            dis_real = discriminator(noise_target, in_text)
            dis_fake = discriminator(noise_out, in_text)
        else:
            dis_real = discriminator(target_poses, in_text)
            dis_fake = discriminator(out_dir_vec.detach(), in_text)
            dis_fake_beat = discriminator(beat_out.detach(), in_text)  

        #dis_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8)))  # ns-gan
        beat_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake_beat + 1e-8)))  # ns-gan
        #all_error = dis_error + beat_error 
        all_error = beat_error 
        all_error.backward()
        dis_optim.step()

    ###########################################################################################
    # train G
    pose_dec_optim.zero_grad()

    # decoding
    out_dir_vec, out_beat, z, z_mu, z_logvar = pose_decoder(pre_seq, time_seq, in_text, in_audio, vid_indices)

    # loss
    beta = 0.1
    huber_loss = F.smooth_l1_loss(out_dir_vec / beta, target_poses / beta) * beta
    beat_huber_loss = F.smooth_l1_loss(out_beat / beta, target_poses / beta) * beta
    
    delta_target = target_poses[:, 1:, :] - target_poses[:, :-1, :] 
    delta_out_dir_vec = out_dir_vec[:, 1:, :] - out_dir_vec[:, :-1, :]
    delta_beat = out_beat[:, 1:, :] - out_beat[:, :-1, :] 
    delta_huber_loss = F.smooth_l1_loss(delta_out_dir_vec / beta, delta_target / beta) * beta
    delta_beat_huber_loss = F.smooth_l1_loss(delta_beat / beta, delta_target / beta) * beta

    dis_output = discriminator(out_dir_vec, in_text)
    beat_output = discriminator(out_beat, in_text)
    gen_error = -torch.mean(torch.log(dis_output + 1e-8))
    beat_error = -torch.mean(torch.log(beat_output + 1e-8))
    kld = div_reg = None

    rand_idx = torch.randperm(vid_indices.shape[0])
    rand_vids = vid_indices[rand_idx]

    out_dir_vec_rand_vid, out_beat_vec_rand_vid, z_rand_vid, _, _ = pose_decoder(pre_seq, time_seq, in_text, in_audio, rand_vids)
    beta = 0.05
    pose_l1 = F.smooth_l1_loss(out_dir_vec / beta, out_dir_vec_rand_vid.detach() / beta, reduction='none') * beta
    pose_l1 = pose_l1.sum(dim=1).sum(dim=1)
    pose_l1 = pose_l1.view(pose_l1.shape[0], -1).mean(1)

    beat_pose_l1 = F.smooth_l1_loss(out_beat / beta, out_beat_vec_rand_vid.detach() / beta, reduction='none') * beta
    beat_pose_l1 = beat_pose_l1.sum(dim=1).sum(dim=1)
    beat_pose_l1 = beat_pose_l1.view(beat_pose_l1.shape[0], -1).mean(1)

    z_l1 = F.l1_loss(z.detach(), z_rand_vid.detach(), reduction='none')
    z_l1 = z_l1.view(z_l1.shape[0], -1).mean(1)

    div_reg = -(pose_l1 / (z_l1 + 1.0e-5))
    div_reg = torch.clamp(div_reg, min=-1000)
    div_reg = div_reg.mean()

    beat_div_reg = -(beat_pose_l1 / (z_l1 + 1.0e-5))
    #beat_div_reg = -(beat_pose_l1 )
    beat_div_reg = torch.clamp(beat_div_reg, min=-1000)
    beat_div_reg = beat_div_reg.mean()

    # speaker embedding KLD
    kld = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    #loss = args.loss_regression_weight * (delta_huber_loss + delta_beat_huber_loss) + args.loss_regression_weight * (huber_loss + beat_huber_loss) + args.loss_kld_weight * kld + args.loss_reg_weight * (div_reg + beat_div_reg) 
    loss = args.loss_regression_weight * delta_beat_huber_loss + args.loss_regression_weight * beat_huber_loss + args.loss_kld_weight * kld #+ args.loss_reg_weight *  beat_div_reg


    if epoch > warm_up_epochs:
        loss += args.loss_gan_weight * beat_error

    #pdb.set_trace()

    loss.backward()
    pose_dec_optim.step()

    ret_dict = {'loss': args.loss_regression_weight * huber_loss.item()}
    if kld:
        ret_dict['KLD'] = args.loss_kld_weight * kld.item()
    if div_reg:
        ret_dict['DIV_REG'] = args.loss_reg_weight * div_reg.item()

    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        ret_dict['gen'] = args.loss_gan_weight * (gen_error + beat_error).item()
        ret_dict['dis'] = all_error.item()
    return ret_dict

