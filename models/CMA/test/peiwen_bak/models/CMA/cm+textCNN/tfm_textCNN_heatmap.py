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
from transformer.Models import Transformer

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "tfm_textCNN_heatmap"
ConType = ["No"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
#          "S18"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
names = ["S18"]

# 所用的数据目录路径
# data_document_path = "../dataset"
data_document_path = "../dataset_16"
# data_document_path = "../dataset_csp"
# data_document_path = "../dataset_fbcsp"
# data_document_path = "../data_all_split"
# data_document_path = "/document/data/eeg/dataset"
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
max_epoch = 60
min_epoch = 0
isEarlyStop = False

# 非常用参数，分别是 被试数量、通道数量、trail数量、trail内数据点数量、测试集比例、验证集比例
# 一般不需要调整
people_number = 18
eeg_channel = 16
audio_channel = 1
channel_number = eeg_channel + audio_channel * 2
trail_number = 20
cell_number = 3500
test_percent = 0.1
vali_percent = 0.1

conv_eeg_channel = 16
conv_audio_channel = 16
conv_time_size = 9
conv_output_channel = 10
fc_number = 30

conv_eeg_audio_number = 3
output_fc_number = conv_eeg_audio_number * conv_output_channel

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
                 attn_mask=False):
        super(CNN, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        self.conv1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(conv_audio_channel, conv_output_channel, 8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        ) for i in range(conv_eeg_audio_number)])

        self.conv2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(conv_audio_channel, conv_output_channel, 9),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        ) for i in range(conv_eeg_audio_number)])

        self.conv3 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(conv_audio_channel, conv_output_channel, 10),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        ) for i in range(conv_eeg_audio_number)])

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(fc_number, fc_number),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(fc_number, conv_output_channel)
        ) for i in range(conv_eeg_audio_number)])

        self.output_fc = nn.Sequential(
            nn.Linear(output_fc_number, output_fc_number), nn.Sigmoid(),
            nn.Linear(output_fc_number, 2), nn.Sigmoid()
        )

        self.transformerA = Transformer(
            # opt.src_vocab_size,
            # opt.trg_vocab_size,
            # src_pad_idx=opt.src_pad_idx,
            # trg_pad_idx=opt.trg_pad_idx,  # 这四个是转换词向量用的
            # trg_emb_prj_weight_sharing=opt.proj_share_weight,  # 最后的fc和decoder embedding共享权重
            # emb_src_trg_weight_sharing=opt.embs_share_weight,  # encoder embedding和decoder embedding共享权重
            d_k=16,  # qk通道数
            d_v=16,  # v通道数
            d_model=16,  # fc2的输入
            d_word_vec=16,  # 特征通道数
            d_inner=16,  # fc2的中间隐藏层
            n_layers=6,  # 层数
            n_head=1,  # head number
            dropout=0.1)  # 所有dropout层

        self.transformerB = Transformer(
            d_k=16,  # qk通道数
            d_v=16,  # v通道数
            d_model=16,  # fc2的输入
            d_word_vec=16,  # 特征通道数
            d_inner=16,  # fc2的中间隐藏层
            n_layers=6,  # 层数
            n_head=1,  # head number
            dropout=0.1)  # 所有dropout层


    def forward(self, x):

        wavA = x[0, 0, 0:1, :]
        wavA = torch.t(wavA).unsqueeze(0)
        eeg = x[0, 0, 1:17, :]
        eeg = torch.t(eeg).unsqueeze(0)
        wavB = x[0, 0, 17:18, :]
        wavB = torch.t(wavB).unsqueeze(0)

        # print(x.shape)
        # print(wavA.shape)
        # print(eeg.shape)
        # print(wavB.shape)

        wavA = torch.zeros(wavA.shape[0], wavA.shape[1], conv_audio_channel).to(device) + wavA
        wavB = torch.zeros(wavA.shape[0], wavB.shape[1], conv_audio_channel).to(device) + wavB

        wavA = self.transformerA(wavA, eeg).unsqueeze(0)
        wavB = self.transformerB(wavB, eeg).unsqueeze(0)

        # print("wavAB")
        # print(wavA.shape)
        # print(wavB.shape)

        wavA = wavA.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        wavB = wavB.transpose(1, 2)

        # textCNN
        data = [wavA, eeg, wavB]
        for i in range(conv_eeg_audio_number):
            # print(wavA.shape)
            temp1 = self.conv1[i](data[i]).view(-1, conv_output_channel)
            # print(wavA1.shape)
            temp2 = self.conv2[i](data[i]).view(-1, conv_output_channel)
            # print(wavA2.shape)
            temp3 = self.conv3[i](data[i]).view(-1, conv_output_channel)
            # print(wavA3.shape)
            temp = torch.cat([temp1, temp2, temp3], dim=1)
            data[i] = self.fc[i](temp)

        x = torch.cat([data[i] for i in range(conv_eeg_audio_number)], dim=1)
        output = self.output_fc(x)
        return output


# 模型权重初始化
def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        try:
            m.weight.data.uniform_(-1.0, 1.0)
            m.bias.data.fill_(0)
        except:
            print("weights_init fail")


# 模型参数和初始化
myNet = CNN()
myNet.apply(weights_init_uniform)
clip = 0.8
optimzer = torch.optim.Adam(myNet.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.randint(4, 7)
device = torch.device('cuda:' + str(0))
# device = torch.device('cpu')
myNet = myNet.to(device)
loss_func = loss_func.to(device)
