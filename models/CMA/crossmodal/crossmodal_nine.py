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
label = "crossmodal_nine"
ConType = ["No"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
names = ["S5"]

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
delay = 0
batch_size = 1
max_epoch = 80
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

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, (65, 9)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 10, (65, 9)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fcn = nn.Sequential(
            nn.Linear(20, 20), nn.Sigmoid(),
            nn.Linear(20, 20), nn.Sigmoid()
        )

        # self.attention_fcn = nn.Sequential(
        #     nn.Linear(94, 94), nn.ReLU(), nn.Dropout(p=0),
        #     nn.Linear(94, 94)
        # )
        # self.attention_output = nn.Sequential(
        #     nn.Linear(94, 1), 
        #     nn.Sigmoid()
        # )

        # self.attention_fcn2 = nn.Sequential(
        #     nn.Linear(94, 94), nn.ReLU(), nn.Dropout(p=0),
        #     nn.Linear(94, 94)
        # )
        # self.attention_output2 = nn.Sequential(
        #     nn.Linear(94, 1), 
        #     nn.Sigmoid()
        # )

        self.proj_images = nn.Conv1d(64, 1, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(64, 1, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(1, 30, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(1, 30, 1, bias=False)

        self.trans_a2e = self.get_network(self_type='a2e')
        self.trans_a2e2 = self.get_network(self_type='a2e')
        self.trans_e2a = self.get_network(self_type='e2a')
        self.trans_e2a2 = self.get_network(self_type='e2a')

        # self.trans_eeg_mem = self.get_network(self_type='e_mem', layers=3)
        # self.trans_audio_mem = self.get_network(self_type='a_mem', layers=3)
        # self.trans_eeg_mem2 = self.get_network(self_type='e_mem', layers=3)
        # self.trans_audio_mem2 = self.get_network(self_type='a_mem', layers=3)

    def get_network(self, self_type='a2e', layers=-1):
        if self_type in ['a2e']:
            embed_dim, num_heads, attn_dropout = 30, 5, 0.1
        elif self_type in ['e2a']:
            embed_dim, num_heads, attn_dropout = 1, 1, 0
        elif self_type == 'e_mem':
            embed_dim, num_heads, attn_dropout = 64, 4, 0.1
        elif self_type == 'a_mem':
            embed_dim, num_heads, attn_dropout = 30, 5, 0.1
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def cross_modal(self, x_audio, x_eeg, is_images=False, conv1D=False):
        x_audio = x_audio.transpose(1, 2)
        self.orig_d_audio=x_audio.shape[1]

        self.d_images_2 = x_eeg.shape[2]

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
        x_audio = x_audio.permute(2, 0, 1)
        x_eeg = x_eeg.permute(2, 0, 1)

        # audio_trans_eeg = self.trans_a2e(proj_x_audio, proj_x_eeg, proj_x_eeg)
        # audio_trans_eeg = x_eeg
        # audio_trans_eeg = self.trans_eeg_mem(audio_trans_eeg)
        # audio_trans_eeg = audio_trans_eeg[-1]
        # audio_trans_eeg = audio_trans_eeg.permute(1,0,2)

        eeg_trans_audio, matrixs = self.trans_e2a(proj_x_eeg, x_audio, x_audio, True)
        # eeg_trans_audio = self.trans_audio_mem(eeg_trans_audio)
        # eeg_trans_audio = eeg_trans_audio[-1]
        eeg_trans_audio = eeg_trans_audio.permute(1,0,2)

        # output = torch.cat([eeg_trans_audio, audio_trans_eeg], dim=1)
        # output = output + self.attention_fcn(output)
        # output = self.attention_output(output)

        # return eeg_trans_audio, audio_trans_eeg
        return eeg_trans_audio, matrixs

    def cross_modal2(self, x_audio, x_eeg, is_images=False, conv1D=False):
        x_audio = x_audio.transpose(1, 2)
        self.d_images_2 = x_eeg.shape[2]
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
        x_audio = x_audio.permute(2, 0, 1)
        x_eeg = x_eeg.permute(2, 0, 1)

        # audio_trans_eeg = self.trans_a2e2(proj_x_audio, proj_x_eeg, proj_x_eeg)
        # audio_trans_eeg = x_eeg
        # audio_trans_eeg = self.trans_eeg_mem2(audio_trans_eeg)
        # audio_trans_eeg = audio_trans_eeg[-1]
        # audio_trans_eeg = audio_trans_eeg.permute(1,0,2)

        eeg_trans_audio, matrixs = self.trans_e2a2(proj_x_eeg, x_audio, x_audio, True)
        # eeg_trans_audio = self.trans_audio_mem2(eeg_trans_audio)
        # eeg_trans_audio = eeg_trans_audio[-1]
        eeg_trans_audio = eeg_trans_audio.permute(1,0,2)

        # output = torch.cat([eeg_trans_audio, audio_trans_eeg], dim=1)
        # output = output + self.attention_fcn2(output)
        # output = self.attention_output2(output)

        # return eeg_trans_audio, audio_trans_eeg
        return eeg_trans_audio, matrixs

    def forward(self, x):
        wavA = x[0, 0, 0:1, :]
        wavA = torch.t(wavA).unsqueeze(0)
        eeg  = x[0, 0, 1:65, :]
        eeg = torch.t(eeg).unsqueeze(0)
        wavB = x[0, 0, 65:66, :]
        wavB = torch.t(wavB).unsqueeze(0)

        wavA, matrixs1 = self.cross_modal(wavA, eeg, is_images=False, conv1D=True)
        wavB, matrixs2 = self.cross_modal2(wavB, eeg, is_images=False, conv1D=True)

        wavA = wavA.transpose(1, 2).unsqueeze(0)
        eeg = eeg.transpose(1, 2).unsqueeze(0)
        wavB = wavB.transpose(1, 2).unsqueeze(0)

        wavA = torch.cat([wavA, eeg], dim=2)
        wavB = torch.cat([wavB, eeg], dim=2)

        # print(wavA.shape)

        wavA = self.conv1(wavA)
        wavB = self.conv1(wavB)

        # print(wavA.shape)

        wavA = wavA.view(-1, 10)
        wavB = wavB.view(-1, 10)

        x = torch.cat([wavA, wavB], dim=1)

        output = self.fcn(x)
        return output, matrixs1, matrixs2

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
device = torch.device('cuda:' + str(0))
# device = torch.device('cpu')
myNet = myNet.to(device)
loss_func = loss_func.to(device)
