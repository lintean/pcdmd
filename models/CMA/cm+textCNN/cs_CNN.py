import torch
import torch.nn as nn
from ecfg import *

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "会议方向_cs"
ConType = ["No"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
# names = ["S5", "S6", "S7"]

# 所用的数据目录路径
data_document_path = origin_data_document + "/dataset"

CNN_file = "./CNN_csp_conference.py"
CNN_split_file = "./CNN_split.py"
data_document = "./data/split_dot_test_cs"

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
overlap = 0.5
window_length = 70 * 2
delay = 0
batch_size = 1
max_epoch = 100
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
isALLTrain = True

# 预训练选择
# 只有train就是单独训练、只有pretrain是跨被试、两者都有是预训练
# 跨被试还需要上方的 isALLTrain 为 True
need_pretrain = True
need_train = False


# 整体模型


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(18, conv_output_channel, 9),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.output_fc = nn.Sequential(
            nn.Linear(conv_output_channel, conv_output_channel), nn.Sigmoid(),
            nn.Linear(conv_output_channel, 2), nn.Sigmoid()
        )

        self.fc = nn.Linear(132, 1)


    def forward(self, x):
        # eeg = x[0, 0, 1:-1, :].unsqueeze(0)
        eeg = x.squeeze(0)
        # print(eeg.shape)

        # CNN
        eeg = self.conv(eeg)
        # eeg = eeg.squeeze(0)
        # print(eeg.shape)
        # eeg = self.fc(eeg)
        # print(eeg.shape)
        eeg = eeg.view(-1, conv_output_channel)
        # print(x.shape)
        output = self.output_fc(eeg)
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
# optimzer = torch.optim.Adam(myNet.parameters(), lr=1e-3)
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)
# optimzer = torch.optim.Adam(myNet.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=0.001)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(gpu_random))
# device = torch.device('cpu')
