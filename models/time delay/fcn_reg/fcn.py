import torch
import torch.nn as nn
from ecfg import *
import eutils.util as util
from dotmap import DotMap
import torch.fft

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_document_path = origin_data_document + "/dataset_csv_Jon_202102B"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "fcn"
args.ConType = ["No"]
# args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
names = ["S1"]

args.CNN_file = "./CNN_normal.py"
args.CNN_split_file = "./CNN_split.py"
args.data_document = "./data_new"

# 加载数据集元数据
data_meta = util.read_json(args.data_document + "/metadata.json") if os.path.isfile(args.data_document + "/metadata.json") \
    else util.read_json(args.data_document_path + "/metadata.json")

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
args.overlap = 26/27
# args.window_length = data_meta.fs * 2
args.window_length = 27
args.delay = 0
args.batch_size = 16
args.max_epoch = 100
args.min_epoch = 0
args.early_patience = 0

# 非常用参数，分别是 被试数量、通道数量、trail数量、trail内数据点数量、测试集比例、验证集比例
# 一般不需要调整
args.people_number = data_meta.people_number
args.eeg_band = data_meta.eeg_band
args.eeg_channel_per_band = data_meta.eeg_channel_per_band
args.eeg_channel = args.eeg_band * args.eeg_channel_per_band
args.audio_band = data_meta.audio_band
args.audio_channel_per_band = data_meta.audio_channel_per_band
args.audio_channel = args.audio_band * args.audio_channel_per_band
args.channel_number = args.eeg_channel + args.audio_channel * 2
args.trail_number = data_meta.trail_number
args.cell_number = data_meta.cell_number
args.bands_number = data_meta.bands_number
args.fs = data_meta.fs
args.test_percent = 0.1
args.vali_percent = 0.1

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
args.isDS = True
# isFM为0是男女信息，为1是方向信息
args.isFM = 0
# 回归模型还是分类模型
args.is_regression = True

# 数据划分选择
# 测试集划分是否跨trail
args.isBeyoudTrail = False
# 是否使用100%的数据作为训练集，isBeyoudTrail=False、isALLTrain=True、need_pretrain = True、need_train = False说明跨被试
args.isALLTrain = False

# 预训练选择
# 只有train就是单独训练、只有pretrain是跨被试、两者都有是预训练
# 跨被试还需要上方的 isALLTrain 为 True
args.need_pretrain = False
args.need_train = True

# grads = {}
#
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(args.eeg_channel * args.window_length, 2), nn.BatchNorm1d(2), nn.Tanh(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        # input shape: (batch, channel, feature, time)
        wavA = x[:, 0, 0:args.audio_channel, :].transpose(1, 2)
        eeg = x[:, 0, args.audio_channel:-args.audio_channel, :].transpose(1, 2)
        wavB = x[:, 0, -args.audio_channel:, :].transpose(1, 2)

        # data shape: (batch, time, feature)
        output = self.fc(eeg.reshape(args.batch_size, -1)).reshape(-1)

        return output

def corr_loss(act, pred):
    cov = torch.mean((act - torch.mean(act)) * (pred - torch.mean(pred)))
    return 1 - (cov / (torch.std(act) * torch.std(pred) + 1e-7))

# 模型参数和初始化
myNet = CNN()
clip = 0.8

lr = 1e-4
optimzer = torch.optim.Adam(myNet.parameters(), lr=lr)
# optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = corr_loss

# 启用gpu
# device = torch.device('cuda:' + str(gpu_random))
device = torch.device('cpu')