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
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "peiwen_140_No_1_FBCSP_e"
ConType = ["No", "Low", "High"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]

# 所用的数据目录路径
# data_document_path = "../dataset"
# data_document_path = "../dataset_csp"
# data_document_path = "../dataset_fbcsp"
# data_document_path = "/document/data/eeg/dataset"
# data_document_path = "/document/data/eeg/dataset_csp"
data_document_path = "/document/data/eeg/dataset_fbcsp"

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
overlap = 0.5
window_length = 140
delay = 7
batch_size = 1
max_epoch = 100
min_epoch = 0
isEarlyStop = True

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
# 是否使用FB
useFB = True
bands_number = 5

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
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
	        nn.Conv2d(1,5,(65,5)),
	        nn.ELU(),
	        nn.AdaptiveAvgPool2d((2,10)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(5,10,(2,5)),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fcn = nn.Sequential(
            nn.Linear(10,10),nn.Sigmoid(),
            nn.Linear(10,2),nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x) 
        
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
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.randint(4, 7)
device = torch.device('cuda:' + str(gpu_random))
# device = torch.device('cpu')
myNet = myNet.to(device)
loss_func = loss_func.to(device)


