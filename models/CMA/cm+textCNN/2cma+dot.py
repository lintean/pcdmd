import torch
import torch.nn as nn
from modules.transformer import TransformerEncoder
from ecfg import *
# from eutils.util import heatmap

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "2cma+dot_e_to_s"
ConType = ["No"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
# names = ["S2"]

# 所用的数据目录路径
data_document_path = origin_data_document + "/dataset"

CNN_file = "./CNN_csp.py"
CNN_split_file = "./CNN_split.py"
data_document = "./data/split_dot_test"

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
eeg_channel = 64
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

conv_eeg_audio_number = 4
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

# grads = {}
#
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

# 整体模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.channel = [16, 16, 16, 16]
        self.ofc_channel = window_length

        self.conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.channel[i], conv_output_channel, 9),
            nn.ReLU(),
        ) for i in range(conv_eeg_audio_number)])

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.conference_CNN = nn.Sequential(
            nn.Conv1d(64 * 4, conv_output_channel, 9),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.output_fc = nn.Sequential(
            nn.Linear(2, 2), nn.Sigmoid(),
            nn.Linear(2, 2), nn.Sigmoid()
        )

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])
        self.average = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        self.proj_images = nn.Conv1d(eeg_channel, 16, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(eeg_channel, 16, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(audio_channel, 16, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(audio_channel, 16, 1, bias=False)

        # self.cm_attn = nn.ModuleList([MultiheadAttention(
        #     embed_dim=16,
        #     num_heads=1,
        #     attn_dropout=0
        # ) for i in range(conv_eeg_audio_number)])

        self.cm_attn = nn.ModuleList([TransformerEncoder(
            embed_dim=16,
            num_heads=1,
            layers=5,
            attn_dropout=0,
            relu_dropout=0,
            res_dropout=0,
            embed_dropout=0,
            attn_mask=False
        ) for i in range(conv_eeg_audio_number)])

        self.lstm_wavA = nn.LSTM(16, 16, 8)
        self.lstm_wavB = nn.LSTM(16, 16, 8)
        self.lstm_eeg = nn.LSTM(16, 16, 8)

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     # CNN
    #     for i in range(conv_eeg_audio_number):
    #         nn.init.uniform_(self.conv[i][0].weight, a=-1, b=1)
    #         nn.init.uniform_(self.conv[i][0].bias, a=-1, b=1)
    #     nn.init.uniform_(self.output_fc[0].weight, a=-1, b=1)
    #     nn.init.uniform_(self.output_fc[0].bias, a=-1, b=1)
    #     nn.init.uniform_(self.output_fc[2].weight, a=-1, b=1)
    #     nn.init.uniform_(self.output_fc[2].bias, a=-1, b=1)
    #
    #     nn.init.uniform_(self.conference_CNN[0].weight, a=-1, b=1)
    #     nn.init.uniform_(self.conference_CNN[0].bias, a=-1, b=1)
    #
    #     # dot
    #     for i in range(2):
    #         nn.init.uniform_(self.fc[i][0].weight, a=-1, b=1)
    #         nn.init.uniform_(self.fc[i][0].bias, a=-1, b=1)
    #
    #     # cma
    #     nn.init.uniform_(self.proj_images.weight, a=-1, b=1)
    #     nn.init.uniform_(self.proj_images2.weight, a=-1, b=1)
    #     nn.init.uniform_(self.proj_audio.weight, a=-1, b=1)
    #     nn.init.uniform_(self.proj_audio2.weight, a=-1, b=1)
    #
    #     # lstm
    #     # 不知道。。

    def dot(self, a, b):
        # 皮尔逊相似度
        # mean_a = torch.mean(a, dim=0)
        # mean_b = torch.mean(b, dim=0)
        # a = a - mean_a
        # b = b - mean_b
        # 余弦
        temp = torch.tensordot(a, b, dims=[[0], [0]])
        temp = torch.diag(temp)
        norm_a = torch.norm(a, p=2, dim=0)
        norm_b = torch.norm(b, p=2, dim=0)
        temp = temp / (norm_a * norm_b)
        return temp

    def Euclidean_Distance(self, a, b):
        # 欧氏距离
        temp = a - b
        temp = torch.norm(temp, p=2, dim=0)
        return temp

    def forward(self, x):
        wavA = x[0, 0, 0:1, :]
        wavA = torch.t(wavA).unsqueeze(0)
        eeg = x[0, 0, 1:-1, :]
        eeg = torch.t(eeg).unsqueeze(0)
        wavB = x[0, 0, -1:, :]
        wavB = torch.t(wavB).unsqueeze(0)

        # wav and eeg shape: (Batch, Time, Channel), wav Channel 1 to 16
        # wavA = self.ChannelWav(wavA.squeeze(2).squeeze(0)).unsqueeze(0)
        # wavB = self.ChannelWav(wavB.squeeze(2).squeeze(0)).unsqueeze(0)
        # eeg = eeg[:, window_length - self.ofc_channel:, :]
        # origin_wavA = wavA.clone().transpose(1, 2)
        # origin_wavB = wavB.clone().transpose(1, 2)

        wavA = wavA.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        wavB = wavB.transpose(1, 2)

        # wav Channel 1 to 16
        # wavA.register_hook(save_grad('1convA'))
        # wavB.register_hook(save_grad('1convB'))
        wavA = self.proj_audio(wavA)
        wavB = self.proj_audio2(wavB)
        # eeg = self.proj_images(eeg)

        # print(wavA.shape)
        # print(eeg.shape)

        # # LSTM input shape: Time x Batch x Channel
        # wavA = wavA.permute(2, 0, 1)
        # wavB = wavB.permute(2, 0, 1)
        # eeg = eeg.permute(2, 0, 1)
        # wavA, _ = self.lstm_wavA(wavA)
        # wavB, _ = self.lstm_wavA(wavB)
        # eeg, _ = self.lstm_eeg(eeg)
        # wavA = wavA.permute(1, 2, 0)
        # wavB = wavB.permute(1, 2, 0)
        # eeg = eeg.permute(1, 2, 0)


        # heatmap(eeg[0])

        # 4CMA
        # multihead_attention Input shape: Time x Batch x Channel
        # wav and eeg shape: (Batch, Channel, Time)
        data = [wavA, eeg, eeg, wavB]
        kv = [eeg, wavA, wavB, eeg]
        weight = [0 for i in range(conv_eeg_audio_number)]
        for j in range(2):
            i = j + 1
            # data[i].register_hook(save_grad('CMA' + str(i)))
            data[i] = data[i].permute(2, 0, 1)
            kv[i] = kv[i].permute(2, 0, 1)
            data[i] = self.cm_attn[i](data[i], kv[i], kv[i])
            data[i] = data[i].permute(1, 2, 0)
            # data[i].register_hook(save_grad('conv' + str(i)))

        # heatmap(data[0][0])

        # dot
        # wav and eeg shape: (Batch, Channel, Time)
        data_dot = None
        for i in range(2):
            temp1 = self.dot(data[i * 3].squeeze(0), data[i + 1].squeeze(0))
            # temp2 = self.Euclidean_Distance(data[i * 3].squeeze(0), data[i + 1].squeeze(0))
            temp1 = self.fc[i](temp1.unsqueeze(1)).squeeze(1)
            temp1 = self.average(temp1.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            # temp2 = self.fc2[i](temp2.unsqueeze(1))
            data_dot = temp1 if data_dot is None else torch.cat([data_dot, temp1], dim=0)
            # data_dot = torch.cat([data_dot, temp2.squeeze(1)], dim=0)
        output = self.output_fc(data_dot).unsqueeze(0)

        # CNN
        # for i in range(conv_eeg_audio_number):
        #     data[i] = self.conv[i](data[i])
        #     # if i == 0:
        #         # data[0].register_hook(save_grad('pool0'))
        #     data[i] = self.pool(data[i]).view(-1, conv_output_channel)
        #
        # x = torch.cat([data[i] for i in range(conv_eeg_audio_number)], dim=1)
        # output = self.output_fc(x)

        # CNN conference
        # data = torch.cat([data[i] for i in range(conv_eeg_audio_number)], dim=1)
        # x = self.conference_CNN(data)
        # x = x.view(-1, conv_output_channel)
        # output = self.output_fc(x)
        return output


# 模型参数和初始化
myNet = CNN()
clip = 0.8

lr = 1e-4
# params = [{'params': myNet.conv.parameters(), 'lr': lr},
#           {'params': myNet.conference_CNN.parameters(), 'lr': lr},
#           {'params': myNet.output_fc.parameters(), 'lr': lr},
#           {'params': myNet.fc.parameters(), 'lr': lr},
#           {'params': myNet.proj_images.parameters(), 'lr': lr},
#           {'params': myNet.proj_images2.parameters(), 'lr': lr},
#           {'params': myNet.cm_attn[0].in_proj_weight, 'lr': lr * 1},
#           {'params': myNet.cm_attn[0].in_proj_bias, 'lr': lr},
#           {'params': myNet.cm_attn[0].out_proj.parameters(), 'lr': lr * 1},
#           {'params': myNet.cm_attn[1].in_proj_weight, 'lr': lr * 1},
#           {'params': myNet.cm_attn[1].in_proj_bias, 'lr': lr},
#           {'params': myNet.cm_attn[1].out_proj.parameters(), 'lr': lr * 1},
#           {'params': myNet.cm_attn[2].in_proj_weight, 'lr': lr * 1},
#           {'params': myNet.cm_attn[2].in_proj_bias, 'lr': lr},
#           {'params': myNet.cm_attn[2].out_proj.parameters(), 'lr': lr * 1},
#           {'params': myNet.cm_attn[3].in_proj_weight, 'lr': lr * 1},
#           {'params': myNet.cm_attn[3].in_proj_bias, 'lr': lr},
#           {'params': myNet.cm_attn[3].out_proj.parameters(), 'lr': lr * 1},
#           {'params': myNet.proj_audio.parameters(), 'lr': lr * 1},
#           {'params': myNet.proj_audio2.parameters(), 'lr': lr * 1},
#           {'params': myNet.lstm.parameters(), 'lr': lr}]
optimzer = torch.optim.Adam(myNet.parameters(), lr=lr)
# optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(gpu_random))