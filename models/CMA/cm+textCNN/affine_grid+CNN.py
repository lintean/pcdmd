import torch
import torch.nn as nn
import torch.nn.functional as F
from ecode.modules_gpu.multihead_attention import MultiheadAttention
from ecfg import *

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "stn+CNN"
ConType = ["No"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
         "S18"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
# names = ["S2"]

# 所用的数据目录路径
data_document_path = origin_data_document + "/dataset_16"

CNN_file = "./CNN_normal.py"
CNN_split_file = "./CNN_split.py"
data_document = "./data/split_1"

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
overlap = 0.5
window_length = 140
delay = 0
batch_size = 1
max_epoch = 100
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
    def __init__(self):
        super(CNN, self).__init__()

        self.channel = [1, 16, 1, 16]
        self.TimeDelaySta = 50
        self.TimeDelayEnd = 300
        self.TimeDelaySta = int(self.TimeDelaySta / 1000 * 70)
        self.TimeDelayEnd = int(self.TimeDelayEnd / 1000 * 70)
        self.TimeDelayNum = self.TimeDelayEnd - self.TimeDelaySta
        self.ofc_channel = window_length - self.TimeDelayEnd

        self.conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.channel[i], conv_output_channel, 9),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        ) for i in range(conv_eeg_audio_number)])

        self.output_fc = nn.Sequential(
            nn.Linear(output_fc_number, output_fc_number), nn.Sigmoid(),
            nn.Linear(output_fc_number, 2), nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
        self.average = nn.AdaptiveAvgPool1d(1)

        self.proj_images = nn.Conv1d(eeg_channel, conv_eeg_channel, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(eeg_channel, conv_eeg_channel, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(audio_channel, conv_audio_channel, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(audio_channel, conv_audio_channel, 1, bias=False)

        self.td_attn = MultiheadAttention(
            embed_dim=self.ofc_channel,
            num_heads=1,
            attn_dropout=0
        )

        self.cm_attn = nn.ModuleList([MultiheadAttention(
            embed_dim=16,
            num_heads=1,
            attn_dropout=0
        ) for i in range(conv_eeg_audio_number)])

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 2 * 33, 33),
            nn.ReLU(True),
            nn.Linear(33, 1)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0], dtype=torch.float))

    # wav shape: (Batch, Time, Channel=1)
    def ChannelWav(self, wav):

        wav = wav.unsqueeze(0)

        # print(self.TimeDelayNum)

        Temp = torch.empty(self.TimeDelayNum, self.ofc_channel).to(device)
        for k in range(self.TimeDelayNum):
            # print(k, end=" ")
            # print(self.ofc_channel + k)
            Temp[k, :] = wav[:, k:self.ofc_channel + k]

        wav = Temp.transpose(0, 1)

        return wav

    # wav and eeg shape: (Batch, Channel, Time)
    def select_max(self, wavA, wavB, eeg, origin_wavA, origin_wavB):
        weightA = torch.bmm(eeg, wavA.transpose(1, 2)).squeeze(0)
        # weightA = F.softmax(weightA.float(), dim=-2).type_as(weightA)
        # print(weightA[:, 0].sum())
        weightA = torch.sum(weightA, 0)
        # print(weightA)

        weightB = torch.bmm(eeg, wavB.transpose(1, 2)).squeeze(0)
        # weightB = F.softmax(weightB.float(), dim=-2).type_as(weightB)
        weightB = torch.sum(weightB, 0)

        # weight = torch.abs(weightA - weightB)
        weight = weightA + weightB

        max_index = torch.max(weight, 0).indices
        # max_index = 0
        wavA = wavA[:, max_index, :]
        wavB = wavB[:, max_index, :]
        return wavA.unsqueeze(0), wavB.unsqueeze(0), weightA.clone(), weightB.clone(), max_index.clone()

    def stn(self, x, eeg, wavA, wavB):
        xs = self.localization(x)
        # print(xs.shape)
        xs = xs.view(-1, 10 * 2 * 33)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 1)
        # print(theta.shape)

        # temp = torch.tensor([
        #     [0, 1, 0]
        # ], dtype=torch.float).to(device)
        # print(temp.shape)
        # theta = torch.cat([theta, temp], dim=0).unsqueeze(0)
        # print(theta.shape)
        # print(theta)
        theta = torch.tensor([
            [1, 0, 0.02 * theta],
            [0, 1, 0]
        ], dtype=torch.float).to(device)
        theta = theta.unsqueeze(0)
        # print(theta)

        grid = F.affine_grid(theta, wavA.size())
        wavA = F.grid_sample(wavA, grid)
        # print(wavA)

        grid = F.affine_grid(theta, wavB.size())
        wavB = F.grid_sample(wavB, grid)
        return eeg, wavA, wavB

    def forward(self, x):

        wavA = x[0, 0, 0:1, :]
        wavA = torch.t(wavA).unsqueeze(0)
        eeg = x[0, 0, 1:17, :]
        eeg = torch.t(eeg).unsqueeze(0)
        wavB = x[0, 0, 17:18, :]
        wavB = torch.t(wavB).unsqueeze(0)

        # wav and eeg shape: (Batch, Time, Channel), wav Channel 1 to 16
        # wavA = self.ChannelWav(wavA.squeeze(2).squeeze(0)).unsqueeze(0)
        # wavB = self.ChannelWav(wavB.squeeze(2).squeeze(0)).unsqueeze(0)
        # eeg = eeg[:, window_length - self.ofc_channel:, :]
        # origin_wavA = wavA.clone().transpose(1, 2)
        # origin_wavB = wavB.clone().transpose(1, 2)
        wavA = wavA.transpose(1, 2)
        wavB = wavB.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        # wavA = self.change_position(wavA, eeg)
        # wavB = self.change_position(wavB, eeg)
        # print(eeg.unsqueeze(0).shape)
        eeg, wavA, wavB = self.stn(x, eeg.unsqueeze(0), wavA.unsqueeze(0), wavB.unsqueeze(0))
        eeg = eeg.squeeze(0)
        wavA = wavA.squeeze(0)
        wavB = wavB.squeeze(0)
        # print(eeg.shape)

        # wav and eeg shape: (Batch, Time, Channel) to (Channel, Batch, Time)
        # wavA = wavA.permute(2, 0, 1)
        # eeg = eeg.permute(2, 0, 1)
        # wavB = wavB.permute(2, 0, 1)

        # 1TD
        # multihead_attention Input shape: Time x Batch x Channel
        # wavA, wA = self.td_attn(query=eeg, key=wavA, value=wavA)
        # wavB, wB = self.td_attn(query=eeg, key=wavB, value=wavB)

        # wav and eeg shape: (Channel, Batch, Time) to (Batch, Channel, Time)
        # wavA = wavA.permute(1, 0, 2)
        # eeg = eeg.permute(1, 0, 2)
        # wavB = wavB.permute(1, 0, 2)

        # wav Channel 16 to 1
        # wavA, wavB, weightA, weightB, max_index = self.select_max(wavA, wavB, eeg, origin_wavA, origin_wavB)

        # wav Channel 1 to 16
        # wavA = self.proj_audio(wavA)
        # wavB = self.proj_audio2(wavB)

        # 4CMA
        # multihead_attention Input shape: Time x Batch x Channel
        # wav and eeg shape: (Batch, Channel, Time)
        data = [wavA, eeg, wavB]
        # kv = [eeg, wavA, wavB, eeg]
        # weight = [0 for i in range(conv_eeg_audio_number)]
        # for i in range(conv_eeg_audio_number):
        #     data[i] = data[i].permute(2, 0, 1)
        #     kv[i] = kv[i].permute(2, 0, 1)
        #     data[i], weight[i] = self.cm_attn[i](query=data[i], key=kv[i], value=kv[i])
        #     data[i] = data[i].permute(1, 2, 0)

        # CNN
        for i in range(conv_eeg_audio_number):
            data[i] = self.conv[i](data[i]).view(-1, conv_output_channel)

        x = torch.cat([data[i] for i in range(conv_eeg_audio_number)], dim=1)
        output = self.output_fc(x)


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

lr = 0.0001
td_params = list(map(id, myNet.td_attn.out_proj.parameters()))
base_params = filter(lambda p: id(p) not in td_params, myNet.parameters())
params = [{'params': base_params},
          {'params': myNet.td_attn.out_proj.parameters(), 'lr': lr * 1}]
optimzer = torch.optim.Adam(params, lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(gpu_random))