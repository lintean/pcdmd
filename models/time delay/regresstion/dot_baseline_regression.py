import torch
import torch.nn as nn
from ecfg import *

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "dot_baseline_regression"
ConType = ["No"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
         "S18"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
# names = ["S5", "S6", "S7"]

# 所用的数据目录路径
data_document_path = origin_data_document + "/dataset_16"

CNN_file = "./CNN_normal.py"
CNN_split_file = "./CNN_split.py"
data_document = "./data/2s_cm4"

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
overlap = 0.5
window_length = 140
delay = 0
batch_size = 1
max_epoch = 200
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
# output_fc_number = conv_eeg_audio_number * conv_output_channel
output_fc_number = 20

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

        self.channel = [1, 16, 1]

        self.conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.channel[i], conv_output_channel, 9),
            nn.ReLU()
        ) for i in range(conv_eeg_audio_number)])

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.output_fc = nn.Sequential(
            nn.Linear(132*2, 132), nn.Sigmoid(),
            nn.Linear(132, 2), nn.Sigmoid()
        )

    def forward(self, x):

        # 注意这里的delay
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

        # wavA = torch.zeros(wavA.shape[0], wavA.shape[1], conv_audio_channel).to(device) + wavA
        # wavB = torch.zeros(wavA.shape[0], wavB.shape[1], conv_audio_channel).to(device) + wavB

        wavA = wavA.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        wavB = wavB.transpose(1, 2)

        # CNN
        data = [wavA, eeg, wavB]
        # for i in range(conv_eeg_audio_number):
        #     data[i] = self.conv[i](data[i])

        data_dot = None
        for i in range(2):
            temp = torch.tensordot(data[i * 2].squeeze(0), data[1].squeeze(0), dims=[[0], [0]])
            temp = torch.diag(temp)
            temp = self.fc[i](temp.unsqueeze(1))
            data_dot = temp.squeeze(1) if data_dot is None else torch.cat([data_dot, temp.squeeze(1)], dim=0)
            # print(data_dot.shape)
        output = self.output_fc(data_dot).unsqueeze(0)
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
clip = 0.8
optimzer = torch.optim.Adam(myNet.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.2, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=10,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(gpu_random))