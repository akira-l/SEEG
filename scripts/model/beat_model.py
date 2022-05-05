import torch
import torch.nn as nn

from model import vocab
import model.embedding_net
from model.tcn import TemporalConvNet

import pdb 





class BeatGenerator(nn.Module): 
    def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings, z_obj=None):
        super().__init__() 

        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.z_obj = z_obj
        self.input_context = args.input_context

        if self.input_context == 'both':
            self.in_size = 32 + 32 + pose_dim + 1  # audio_feat + text_feat + last pose + constraint bit
        elif self.input_context == 'none':
            self.in_size = pose_dim + 1
        else:
            self.in_size = 32 + pose_dim + 1  # audio or text only



        if self.z_obj:
            self.z_size = 16
            self.in_size += self.z_size
            if isinstance(self.z_obj, vocab.Vocab):
                self.speaker_embedding = nn.Sequential(
                    nn.Embedding(z_obj.n_words, self.z_size),
                    nn.Linear(self.z_size, self.z_size)
                )
                self.speaker_mu = nn.Linear(self.z_size, self.z_size)
                self.speaker_logvar = nn.Linear(self.z_size, self.z_size)
            else:
                pass  # random noise

        self.time_step = 34 
        self.beat_size = 10 * 3 * 3 
        self.side_size = 10 * 3
        self.hidden_size = 128 
        self.gru = nn.GRU(self.hidden_size + 16, # + self.hidden_size//2, 
                          hidden_size=self.hidden_size, 
                          num_layers=4, 
                          batch_first=True, 
                          bidirectional=True, 
                          dropout=0.5) 

        self.beat_emb = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size // 2), 
                nn.LeakyReLU(True), 
              )
                          
        self.pose_emb = nn.Sequential(
                nn.Linear(28, 16), 
                nn.LeakyReLU(True), 
              )

        self.cat_emb = nn.Sequential(
                nn.Linear(self.hidden_size + 16, self.hidden_size), 
                nn.LeakyReLU(True), 
              ) 

        self.out_emb = nn.Sequential(
                nn.Linear(self.hidden_size + 16, 
                          self.hidden_size), 
                nn.LeakyReLU(True), 
              ) 

        self.beat_out_emb = nn.Sequential( 
                nn.Linear(self.hidden_size, 64), 
                nn.LeakyReLU(True), 
                nn.Linear(64, 27), 
              )

        
        self.square_layer = nn.Sequential(
                nn.Linear(3*self.side_size, self.hidden_size),  
                nn.LeakyReLU(True), 
              ) 

        self.oenv_layer = nn.Sequential(
                nn.Linear(3*self.side_size, self.hidden_size), 
                nn.LeakyReLU(True), 
              )

        self.re_sum_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), 
                nn.LeakyReLU(True), 
              )


        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True


    def forward(self, pre_seq, beats, in_text, in_audio, vid_indices=None):

        assert vid_indices is not None
        z_context = self.speaker_embedding(vid_indices)
        z_mu = self.speaker_mu(z_context)
        z_logvar = self.speaker_logvar(z_context)
        z_context = model.embedding_net.reparameterize(z_mu, z_logvar)
 

        bs, _, _ = beats.size()
        gather_out = [] 
        gather_feat = [] 
        for ts in range(self.time_step): 

            if self.do_flatten_parameters:
                self.gru.flatten_parameters()

            if ts == 0: 
                sub_beats = torch.cat([torch.zeros(bs, 2, self.side_size).cuda(), 
                                       beats[:, :, :2*self.side_size]], 2) 
                decoder_hidden = None 
            elif ts == self.time_step - 1: 
                sub_beats = torch.cat([beats[:, :, -2*self.side_size:], 
                                       torch.zeros(bs, 2, self.side_size).cuda()], 2) 
            else: 
                sub_beats = beats[:, :, self.side_size*(ts-1):self.side_size*(ts+2)] 

            squ_out = self.square_layer(sub_beats[:, 0, :]) 
            oenv_out = self.oenv_layer(sub_beats[:, 1, :]) 
            sum_out = squ_out + oenv_out 
            #re_sum_out = self.re_sum_layer(sum_out) 

            pose_out = self.pose_emb(pre_seq[:, ts, :])  
            cat_out = torch.cat([pose_out, sum_out], 1) 
            cat_emb = self.cat_emb(cat_out) 

            output, decoder_hidden = self.gru(cat_out.unsqueeze(1), decoder_hidden) 
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

            cat_id = torch.cat([output, z_context.unsqueeze(1)], 2)  
            #output = self.beat_emb(output) 

            #pose_out = self.pose_emb(pre_seq[:, ts, :]).unsqueeze(1)  
            #gather_feat = torch.cat([output, pose_out], 2) 
            #fin_out = self.out_emb(gather_feat) 

            feat_out = self.out_emb(cat_id) 
            gather_feat.append(feat_out) 
            beat_fin_out = self.beat_out_emb(feat_out)  
            gather_out.append(beat_fin_out) 


        #if return_feat: 
        #    return torch.cat(gather_out, 1), torch.cat(gather_feat, 1), z_context, z_mu, z_logvar
        #else: 
        return torch.cat(gather_out, 1), z_context, z_mu, z_logvar
        #return torch.cat(gather_out, 1), z_context, z_mu, z_logvar








class BeatDiscriminator(nn.Module):
    def __init__(self, args, input_size, n_words=None, word_embed_size=None, word_embeddings=None):
        super().__init__()
        self.input_size = input_size

        if n_words and word_embed_size:
            self.text_encoder = TextEncoderTCN(n_words, word_embed_size, word_embeddings)
            input_size += 32
        else:
            self.text_encoder = None

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_size, num_layers=args.n_layers, bidirectional=True,
                          dropout=args.dropout_prob, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(args.n_poses, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        if self.text_encoder:
            text_feat_seq, _ = self.text_encoder(in_text)
            poses = torch.cat((poses, text_feat_seq), dim=2)

        output, decoder_hidden = self.gru(poses, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output




class BeatConvDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )

        self.gru = nn.GRU(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(28, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        poses = poses.transpose(1, 2)
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)

        output, decoder_hidden = self.gru(feat, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output






class BeatLoopDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.hidden_size = 64
        self.time_step = 34 
        self.side_size = 1 

        self.gru = nn.GRU(self.input_size, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)

        self.reout_layer = nn.Linear(self.hidden_size, 8) 
        #self.fin_out = nn.Sequential(
        #        nn.Linear(34*16, 64),  
        #        nn.LeakyReLU(True),
        #        nn.Linear(64, 1) 
        #        )
        self.fin_out = nn.Linear(34*8, 1) 

        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )

        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(28, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):

        bs = poses.size(0) 
        gather_out = []   
        for ts in range(self.time_step): 

            if self.do_flatten_parameters:
                self.gru.flatten_parameters()

            if ts == 0: 
                sub_pose = torch.cat([torch.zeros(bs, self.side_size, self.input_size).cuda(), 
                                      poses[:, :2*self.side_size, :]], 1) 
                decoder_hidden = None 
            elif ts == self.time_step - 1: 
                sub_pose = torch.cat([poses[:, -2*self.side_size:, :], 
                                       torch.zeros(bs, self.side_size, self.input_size).cuda()], 1) 
            else: 
                sub_pose = poses[:, self.side_size*(ts-1):self.side_size*(ts+2), :] 

            sub_pose = sub_pose.mean(1) 
            output, decoder_hidden = self.gru(sub_pose.unsqueeze(1), decoder_hidden) 
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
            reout = self.reout_layer(output) 
            gather_out.append(reout) 
        gather_out_feat = torch.cat(gather_out, 1) 
        gather_out_view = gather_out_feat.view(bs, -1) 
        fin_out = self.fin_out(gather_out_view) 
        return torch.sigmoid(fin_out) 





