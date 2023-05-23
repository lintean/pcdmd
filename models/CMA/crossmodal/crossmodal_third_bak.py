from scipy.signal import hilbert
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torch.autograd import Function
from torch.autograd import Variable
import math
import random
import sys
import os
from modules.transformer import TransformerEncoder

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "crossmodal_third"
ConType = ["No"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
# names = ["S2"]

# 所用的数据目录路径
# data_document_path = "../dataset"
# data_document_path = "../dataset_csp"
# data_document_path = "../dataset_fbcsp"
# data_document_path = "../data_all_split"
data_document_path = "/document/data/eeg/dataset"
# data_document_path = "/document/data/eeg/dataset_csp"
# data_document_path = "/document/data/eeg/dataset_fbcsp"
# data_document_path = "/document/data/eeg/dataset_fbcsp10"
# data_document_path = "/document/data/eeg/data_all_split"

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
overlap = 0.5
window_length = 140
delay = 7
batch_size = 1
max_epoch = 50
min_epoch = 0
isEarlyStop = False

# 非常用参数，分别是 被试数量、通道数量、trail数量、trail内数据点数量、测试集比例、验证集比例
# 一般不需要调整
people_number = 18
channle_number = 64 + 2
trail_number = 20
cell_number = 3500
test_percent = 0.1
vali_percent = 0.1

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
isDS = True
# isFM为0是男女信息，为1是方向信息
isFM = 0
# 频带数量
bands_number = 1
useFB = False

# 数据划分选择
# 测试集划分是否跨trail
isBeyoudTrail = False
# 是否使用100%的数据作为训练集，isBeyoudTrail=False、isALLTrain=True、need_pretrain = True、need_train = False说明跨被试
isALLTrain = False

# 预训练选择
# 只有train就是单独训练、只有pretrain是跨被试、两者都有是预训练
# 跨被试还需要上方的 isALLTrain 为 True
need_pretrain = False
need_train = True

# 整体模型


class CNN(nn.Module):
    def __init__(self, embed_dim=30,
                 num_heads=5,
                 layers=5,
                 attn_dropout=0.1,
                 relu_dropout=0.1,
                 res_dropout=0.1,
                 embed_dropout=0.25,
                 attn_mask=True):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, (60, 9)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = nn.Sequential(
            nn.Linear(10, 10), nn.Sigmoid(),
            nn.Linear(10, 2), nn.Sigmoid()
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

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

        x_audio = x_audio.transpose(1, 2)
        self.orig_d_audio=x_audio.shape[1]
        self.d_audio=30
        self.d_eeg=30
        self.proj_audio = nn.Conv1d(self.orig_d_audio, self.d_audio, kernel_size=1, padding=0, bias=False).to(device)

        #x_eeg是图片的情况
        if is_images==True:

            # x_eeg原本为[batch_size, seq_len, height, width], 增加channel维度,目前程序只支持channel=1
            if len(x_eeg.shape)==4:
                x_eeg = x_eeg[:, :, np.newaxis, :, :]

            self.images_channel=x_eeg.shape[2]
            pixel_h=x_eeg.shape[3]
            pixel_w=x_eeg.shape[4]
            # 将eegImages卷积成一维
            self.d_images_out1 = 8
            self.d_images_out2 = 1
            self.images_kernel_size1 = (int(pixel_h/2), int(pixel_w/2))
            self.images_kernel_size2 = (pixel_h+1-int(pixel_h/2), int(pixel_w/4))
            self.orig_d_images=self.images_channel

            self.proj_images_1 = nn.Conv2d(self.orig_d_images, self.d_images_out1, self.images_kernel_size1).to(device)
            self.proj_images_2 = nn.Conv2d(self.d_images_out1, self.d_images_out2, self.images_kernel_size2).to(device)

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
            x_eeg = torch.tensor(temp_x_eeg, dtype=torch.float32).to(device)

            self.d_images_2 = temp_channel_len
        else:
            self.d_images_2=x_eeg.shape[2]

        self.proj_images = nn.Conv1d(self.d_images_2, self.d_eeg, kernel_size=1, padding=0, bias=False).to(device)

        x_eeg = x_eeg.transpose(1, 2)

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

        self.trans_a2e = self.get_network(self_type='a2e').to(device)
        audio_trans_eeg = self.trans_a2e(proj_x_audio, proj_x_eeg, proj_x_eeg)
        # audio_trans_eeg = proj_x_eeg

        self.trans_e2a = self.get_network(self_type='e2a').to(device)
        eeg_trans_audio = self.trans_e2a(proj_x_eeg,proj_x_audio,proj_x_audio)
        # eeg_trans_audio = proj_x_audio

        audio_trans_eeg=audio_trans_eeg.permute(1,0,2)
        eeg_trans_audio=eeg_trans_audio.permute(1,0,2)
        return audio_trans_eeg, eeg_trans_audio

    def forward(self, x):
        # print(x.shape)
        wavA = x[0, 0, 0:1, :]
        wavA = torch.t(wavA).unsqueeze(0)
        eeg  = x[0, 0, 1:65, :]
        eeg = torch.t(eeg).unsqueeze(0)

        # print(wavA.shape)
        # print(eeg.shape)

        audio, eeg = self.cross_modal(wavA, eeg, is_images=False, conv1D=True)
        x = torch.cat([audio,eeg],dim=2)
        x = x.transpose(1, 2)
        x = x.unsqueeze(0)

        x = self.conv(x)

        x = x.view(-1, 10)
        output = self.fcn(x)
        return output

# 模型权重初始化


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)


# 模型参数和初始化
myNet = CNN()
myNet.apply(weights_init_uniform)
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.5, weight_decay=0.0000001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.randint(5, 7)
device = torch.device('cuda:' + str(gpu_random))
# device = torch.device('cpu')
myNet = myNet.to(device)
loss_func = loss_func.to(device)
