import torch
import torch.nn as nn

from model import vocab
import model.embedding_net
from model.tcn import TemporalConvNet

import pdb 

class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0






class BeatGenerator(nn.Module): 
    def __init__(self, ): 
        super().__init__() 

        self.time_step = 34 
        self.beat_size = 10 * 3 * 3 
        self.side_size = 10 * 3
        self.hidden_size = 128 
        self.gru = nn.GRU(self.hidden_size + self.hidden_size//2, 
                          hidden_size=self.hidden_size, 
                          num_layers=4, 
                          batch_first=True, 
                          bidirectional=True, 
                          dropout=0.2) 
        self.beat_emb = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size // 2), 
                nn.LeakyReLU(True), 
              )
                          
        self.pose_emb = nn.Sequential(
                nn.Linear(28, self.hidden_size // 2), 
                nn.LeakyReLU(True), 
              )
        self.out_emb = nn.Sequential(
                nn.Linear(self.hidden_size*2, 64), 
                nn.LeakyReLU(True), 
                nn.Linear(64, 27), 
              )
        self.square_layer = nn.Linear(3*self.side_size, self.hidden_size) 
        self.oenv_layer = nn.Linear(3*self.side_size, self.hidden_size) 



    def forward(self, pre_seq, beats): 
        bs, _, _ = beats.size()
        gather_out = [] 
        for ts in range(self.time_step): 
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

            pose_out = self.pose_emb(pre_seq[:, ts, :])  
            cat_out = torch.cat([pose_out, sum_out], 1) 

            output, decoder_hidden = self.gru(cat_out.unsqueeze(1), decoder_hidden) 

            #output = self.beat_emb(output) 

            #pose_out = self.pose_emb(pre_seq[:, ts, :]).unsqueeze(1)  
            #gather_feat = torch.cat([output, pose_out], 2) 
            #fin_out = self.out_emb(gather_feat) 

            fin_out = self.out_emb(output) 
            gather_out.append(fin_out) 


        return torch.cat(gather_out, 1) 













class PoseGenerator(nn.Module):
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

        self.beat_encoder = BeatGenerator() 
        self.audio_encoder = WavEncoder()
        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                           dropout=args.dropout_prob)

        self.speaker_embedding = None
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

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layers, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size//2, pose_dim)
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, pre_seq, time_seq, in_text, in_audio, vid_indices=None):

        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if self.input_context != 'none':
            # audio
            audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)

            # text
            text_feat_seq, _ = self.text_encoder(in_text)
            assert(audio_feat_seq.shape[1] == text_feat_seq.shape[1])

        # z vector; speaker embedding or random noise
        if self.z_obj:
            if self.speaker_embedding:
                assert vid_indices is not None
                z_context = self.speaker_embedding(vid_indices)
                z_mu = self.speaker_mu(z_context)
                z_logvar = self.speaker_logvar(z_context)
                z_context = model.embedding_net.reparameterize(z_mu, z_logvar)
            else:
                z_mu = z_logvar = None
                z_context = torch.randn(in_text.shape[0], self.z_size, device=in_text.device)
        else:
            z_mu = z_logvar = None
            z_context = None

        if self.input_context == 'both':
            in_data = torch.cat((pre_seq, audio_feat_seq, text_feat_seq), dim=2)
        elif self.input_context == 'audio':
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        elif self.input_context == 'text':
            in_data = torch.cat((pre_seq, text_feat_seq), dim=2)
        elif self.input_context == 'none':
            in_data = pre_seq
        else:
            assert False

        if z_context is not None:
            repeated_z = z_context.unsqueeze(1)
            repeated_z = repeated_z.repeat(1, in_data.shape[1], 1)
            in_data = torch.cat((in_data, repeated_z), dim=2)

        #output, decoder_hidden = self.gru(in_data, decoder_hidden)
        #output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        #output = self.out(output.reshape(-1, output.shape[2]))
        #decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)

        beat_gesture = self.beat_encoder(pre_seq, time_seq) 

        #return decoder_outputs+beat_gesture, beat_gesture, z_context, z_mu, z_logvar
        return beat_gesture, beat_gesture, z_context, z_mu, z_logvar



class Discriminator(nn.Module):
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


class ConvDiscriminator(nn.Module):
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
