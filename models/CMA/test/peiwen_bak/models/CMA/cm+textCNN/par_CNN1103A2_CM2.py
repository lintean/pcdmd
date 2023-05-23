import torch
import torch.nn as nn
from ecfg import *

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "par_CNN1103A2_CM3"
print(label)
ConType = ["No"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
         "S18"]

# 所用的数据目录路径
data_document_path = origin_data_document + "/dataset_16"

CNN_file = "./CNN_normal.py"
CNN_split_file = "./CNN_split.py"
data_document = "./data/2s_temp"

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
overlap = 0.5
window_length = 140
winLen = 120
delay = 0
TD_Sta = 5
TD = 10
TD_End = TD_Sta + TD
batch_size = 1
max_epoch = 50
min_epoch = 0
isEarlyStop = False

# 非常用参数，分别是 被试数量、通道数量、trail数量、trail内数据点数量、测试集比例、验证集比例
# 一般不需要调整
people_number = 18
channel_number = 16 + 2
trail_number = 20
cell_number = 3500
test_percent = 0.2
vali_percent = 0.2

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
isDS = True
# isFM为0是男女信息，为1是方向信息
isFM = 0
# 是否使用FB
useFB = True
bands_number = 1

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
        # CM1
        self.CM1fcnQ = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM1fcnKin = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM1fcnVin = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM1fcnQKVout = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM1softMax = nn.Softmax2d()

        # CM2
        self.CM2fcnQ = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM2fcnKin = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM2fcnVin = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM2fcnQKVout = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU6(),
        )
        self.CM2softMax = nn.Softmax2d()

        # 分类器
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 5, (2, 3), stride=(2, 1)),
            nn.ReLU6(),
            nn.Conv2d(5, 5, (2, 4), stride=(2, 4)),
            nn.ReLU6(),
            nn.Conv2d(5, 5, (2, 4), stride=(2, 4)),
            nn.ReLU6(),
            nn.Conv2d(5, 5, (2, 4), stride=(2, 3)),
            nn.ReLU6(),

            # nn.Conv2d(1,1,(16,14),stride=(16,14)),
            # nn.AdaptiveMaxPool2d((3,10)),
        )
        self.fcn = nn.Sequential(
            nn.Linear(30, 15),
            nn.Sigmoid(),
            nn.Linear(15, 2),
            nn.Softmax(dim=1),
        )
        self.proj_audio = nn.Conv1d(1, 16, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(1, 16, 1, bias=False)
    def wavProcess(self,wav,eeg):
        # eeg = torch.ones(eeg.shape).to(device)
        wav = wav * eeg
        wav = wav.squeeze(0).squeeze(0)
        eeg = eeg.squeeze(0).squeeze(0).transpose(0, 1)
        mat = torch.mm(eeg, wav)
        wav = torch.mm(wav, mat)

        return wav.unsqueeze(0).unsqueeze(0)

        # 定义CM模型
    def CM1(self, wav, eeg):
        # print(wav.shape)
        # wav = self.proj_audio(wav).unsqueeze(0)
        wav = self.wavProcess(wav, eeg)
        # print(wav.shape)

        wav = wav.squeeze(0).squeeze(0).transpose(0,1)
        eeg = eeg.squeeze(0).squeeze(0).transpose(0,1)

        Q = self.CM1fcnQ(eeg)
        K = self.CM1fcnKin(wav)
        V = self.CM1fcnVin(wav)
        wav = torch.mm(Q, K.transpose(0, 1))
        wav = self.CM1softMax(wav.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        wav = torch.mm(wav, V)
        wav = self.CM1softMax(wav.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        wav = self.CM1fcnQKVout(wav)
        # wav = wav.transpose(0,1).unsqueeze(0)

        return wav.transpose(0, 1).unsqueeze(0).unsqueeze(0)

    def CM2(self, wav, eeg):
        # wav = self.proj_audio2(wav).unsqueeze(0)
        wav = self.wavProcess(wav, eeg)

        wav = wav.squeeze(0).squeeze(0).transpose(0, 1)
        eeg = eeg.squeeze(0).squeeze(0).transpose(0, 1)

        Q = self.CM2fcnQ(eeg)
        K = self.CM2fcnKin(wav)
        V = self.CM2fcnVin(wav)
        wav = torch.mm(Q, K.transpose(0, 1))
        wav = self.CM2softMax(wav.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        wav = torch.mm(wav, V)
        wav = self.CM2softMax(wav.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        wav = self.CM2fcnQKVout(wav)
        # wav = wav.transpose(0, 1).unsqueeze(0)

        return wav.transpose(0, 1).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        eeg  = x[:, :, 1:-1, :]
        wavA = x[:, :,    0, :]
        wavB = x[:, :,   -1, :]

        # # prepare the wav data and the result is good
        # wavA = self.wavProcess(wavA,eeg)
        # wavB = self.wavProcess(wavB,eeg)

        wavA = self.CM1(wavA, eeg)
        wavB = self.CM2(wavB, eeg)

        y = torch.cat([wavA, eeg, wavB], dim=2)

        y = self.conv1(y).view(-1,30)
        output = self.fcn(y)

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
lrRate = 0.01
optimzer = torch.optim.SGD([
    # {'params':myNet.parameters()},
                            {'params':myNet.conv1.parameters()},{'params':myNet.fcn.parameters()},
                            {'params':myNet.CM1fcnQ.parameters(),'lr':lrRate*1e3},{'params':myNet.CM1fcnKin.parameters(),'lr':lrRate*1e3},
                            {'params':myNet.CM2fcnQ.parameters(),'lr':lrRate*1e3},{'params':myNet.CM2fcnKin.parameters(),'lr':lrRate*1e3},
                            {'params':myNet.CM1fcnVin.parameters(),'lr':lrRate*1e2},{'params':myNet.CM1fcnQKVout.parameters(),'lr':lrRate*1e1},
                            {'params':myNet.CM2fcnVin.parameters(),'lr':lrRate*1e2},{'params':myNet.CM2fcnQKVout.parameters(),'lr':lrRate*1e1}],
                            lr=lrRate, weight_decay=0.0000001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='min', factor=0.5, patience=5, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0,
                                                       eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(gpu_random))
# device = torch.device('cpu')