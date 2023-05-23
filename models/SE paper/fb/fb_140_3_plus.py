from scipy.signal import hilbert
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torch.autograd import Function
from torch.autograd import Variable
import math
import random
import sys
import os

# 使用的参数
# 输入数据选择
label = "fb_140_3_plus"
ConType = ["No"]
names = ["S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]

# data_document_name = "dataset"
data_document_name = "dataset_fbcsp"

# 模型参数
# 重复率
overlap = 0.5
window_length = 140
delay = 7
max_epoch = 1200
# max_epoch = 10000
batch_size = 1
isEarlyStop = True

people_number = 18
channle_number = 64 + 2
trail_number = 20
cell_number = 3500
# cell_number = 3500
test_percent = 0.1
vali_percent = 0.1

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
isDS = True
# isFM为0是男女信息，为1是方向信息
isFM = 0

# 数据划分选择
# 测试集划分是否跨trail
isBeyoudTrail = False
# 是否使用100%的数据作为训练集，isALLTrain=True、need_pretrain = True、need_train = False说明跨被试
isALLTrain = True

# 预训练选择
need_pretrain = True
need_train = False


# 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, (66, 9)),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(10, 10),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.fcn = nn.Sequential(
            nn.Linear(10, 10), nn.Sigmoid(),
            nn.Linear(10, 2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)

        # x = x.view(10,-1)
        # x = torch.t(x)
        # print(x.shape)
        # x = self.attention(x)
        # x = torch.t(x)
        # x = x.view(1, 10, 1, -1)
        x = self.pool(x)

        x = x.view(-1, 10)
        output = self.fcn(x)
        return output


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)


# 模型参数和初始化
myNet = CNN()
myNet.apply(weights_init_uniform)
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.randint(5, 7)
device = torch.device('cuda:' + str(gpu_random))
# device = torch.device('cpu')
myNet = myNet.to(device)
loss_func = loss_func.to(device)


