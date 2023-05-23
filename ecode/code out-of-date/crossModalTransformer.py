import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from modules.transformer import TransformerEncoder

class CrossModalTransformer(nn.Module):

    def __init__(self,embed_dim=30,
                                  num_heads=5,
                                  layers=5,
                                  attn_dropout=0.1,
                                  relu_dropout=0.1,
                                  res_dropout=0.1,
                                  embed_dropout=0.25,
                                  attn_mask=True):
        super(CrossModalTransformer, self).__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.layers=layers
        self.attn_dropout=attn_dropout
        self.relu_dropout=relu_dropout
        self.res_dropout=res_dropout
        self.embed_dropout=embed_dropout
        self.attn_mask=attn_mask

    def get_network(self, self_type='a2e', layers=-1):
        if self_type in ['a2e']:
            embed_dim = self.d_eeg
        elif self_type in ['e2a']:
            embed_dim = self.d_audio
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def cross_modal(self, x_audio, x_eeg, is_images=False, conv1D=False):
        '''
        audio and eeg should have dimension [batch_size, seq_len, n_features]
        if x_eeg is eegImages, the dimension should be [batch_size, seq_len, channel, height, width], default the images channel=1
        x_audio and x_eeg are both numpy array.
        '''

        x_audio = x_audio.transpose((1, 2))
        self.orig_d_audio=x_audio.shape[2]
        self.d_audio=30
        self.d_eeg=30
        self.proj_audio = nn.Conv1d(self.orig_d_audio, self.d_audio, kernel_size=1, padding=0, bias=False)

        #x_eeg是图片的情况
        if is_images==True:

            # x_eeg原本为[batch_size, seq_len, height, width], 增加channel维度
            if x_eeg.shape==4:
                x_eeg = x_eeg[:, :, np.newaxis, :, :]

            self.images_channel=x_eeg.shape[2]
            # 将eegImages卷积成一维
            self.images_kernel_size1 = (16, 16,self.images_channel)
            self.images_kernel_size2 = (17, 8)
            self.d_images_out1 = 8
            self.d_images_out2 = 1
            self.proj_images_1 = nn.Conv2d(self.orig_d_images, self.d_images_out1, self.images_kernel_size1)
            self.proj_images_2 = nn.Conv2d(self.d_images_out1, self.d_images_out2, self.images_kernel_size2)

            temp_batch_size = x_eeg.shape[0]
            temp_x_eeg = []
            temp_seq_len = 0
            temp_channel_len = 0
            for b_index in range(temp_batch_size):
                temp_batch_x_eeg = self.proj_images_1(x_eeg[b_index])
                temp_batch_x_eeg = self.proj_images_2(temp_batch_x_eeg)
                temp_seq_len = temp_batch_x_eeg.shape[0]
                temp_channel_len = temp_batch_x_eeg.shape[3]
                temp_x_eeg.append(temp_batch_x_eeg.cpu().detach().numpy())
            temp_x_eeg = np.array(temp_x_eeg)
            temp_x_eeg = temp_x_eeg.reshape(-1, temp_seq_len, temp_channel_len)
            # x_eeg = torch.from_numpy(temp_x_eeg).cuda()
            x_eeg=temp_x_eeg

            self.d_images_2 = temp_channel_len
        else:
            self.d_images_2=x_eeg.shape[2]

        self.proj_images = nn.Conv1d(self.d_images_2, self.d_eeg, kernel_size=1, padding=0, bias=False)

        x_eeg = x_eeg.transpose((1, 2))

        if conv1D==True:
            proj_x_audio=self.proj_audio(x_audio)
            proj_x_eeg=self.proj_images(x_eeg)
        else:
            proj_x_audio=x_audio
            proj_x_eeg=x_eeg
            self.d_audio=proj_x_audio.shape[1]
            self.d_eeg=proj_x_eeg.shape[1]

        proj_x_audio = proj_x_audio.permute(2, 0, 1)
        proj_x_eeg = proj_x_eeg.permute(2, 0, 1)

        self.trans_a2e=self.get_network(self_type='a2e')
        audio_trans_eeg = self.trans_a2e(proj_x_audio, proj_x_eeg, proj_x_eeg)

        self.trans_e2a=self.get_network(self_type='e2a')
        eeg_trans_audio=self.trans_e2a(proj_x_eeg,proj_x_audio,proj_x_audio)

        return audio_trans_eeg, eeg_trans_audio


