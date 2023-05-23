import torch.nn as nn
import math
import os
import eutils.util as util
from dotmap import DotMap
import torch.fft

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_document_path = device_to_use.origin_data_document + "/KUL_band20210124"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "fcn_multiple_multiple"
args.ConType = ["No"]

args.CNN_file = "./CNN_normal.py"
args.CNN_split_file = "./CNN_split.py"
args.data_document = device_to_use.splited_data_document + "/KUL_multiple_multiple"

# 加载数据集元数据
data_meta = util.read_json(args.data_document + "/metadata.json") if os.path.isfile(args.data_document + "/metadata.json") \
    else util.read_json(args.data_document_path + "/metadata.json")
args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
# names = ["S1"]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
args.overlap = 0
args.window_length = data_meta.fs * 2
# args.window_length = 27
args.delay = 0
args.batch_size = 1
args.max_epoch = 100
args.min_epoch = 0
args.early_patience = 0

# 可视化选项 列表为空表示不希望可视化
args.visualization_epoch = [0, 30, 60, 99]
args.visualization_window_index = [0, 10, 100, 1000]

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
args.is_regression = False

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


# 整体模型


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_output_channel = 10
        self.conv_eeg_audio_number = 1
        self.output_fc_number = self.conv_eeg_audio_number * self.conv_output_channel

        self.conv = nn.Sequential(
            nn.Conv2d(1, math.floor(self.conv_output_channel/3), (3, 3)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(math.floor(self.conv_output_channel/3), math.floor(self.conv_output_channel/3*2), (3, 3)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(math.floor(self.conv_output_channel/3*2), self.conv_output_channel, (3, 3)),
            nn.ReLU(),
        )
        self.pool2d = nn.AdaptiveAvgPool2d(1)

        self.conference_CNN = nn.Sequential(
            nn.Conv1d(args.channel_number, self.conv_output_channel, 9),
            nn.ReLU(),
        )
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        self.output_fc = nn.Sequential(
            nn.Linear(self.output_fc_number, self.output_fc_number), nn.Sigmoid(),
            nn.Linear(self.output_fc_number, 2), nn.Sigmoid()
        )

        self.fc = nn.Linear(132, 1)


    def forward(self, x):
        visualization_weights = []
        # eeg = x[0, 0, audio_channel:-audio_channel, :].unsqueeze(0)
        # eeg = torch.cat([x[0, 0, :audio_channel, :], x[0, 0, -audio_channel:, :]], dim=0).unsqueeze(0)
        # eeg = x
        eeg = x.squeeze(0)
        # print(eeg.shape)

        # CNN
        # eeg = self.conv(eeg)
        # eeg = self.conv2(eeg)
        # eeg = self.conv3(eeg)
        # visualization_weights.append(DotMap(data=eeg[0, 0].clone(), title="conv"))
        # eeg = self.pool2d(eeg)
        eeg = self.conference_CNN(eeg)
        # visualization_weights.append(DotMap(data=eeg[0].clone(), title="conv"))
        eeg = self.pool1d(eeg)
        # eeg = eeg.squeeze(0)
        # print(eeg.shape)
        # eeg = self.fc(eeg)
        # print(eeg.shape)
        eeg = eeg.view(-1, self.conv_output_channel)
        # print(x.shape)
        visualization_weights.append(DotMap(data=eeg.clone(), title="pool"))
        output = self.output_fc(eeg)
        return output, visualization_weights


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
# optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)
optimzer = torch.optim.Adam(myNet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='min', factor=0.5, patience=300, verbose=True, threshold=1e-5, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-8)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
# device = torch.device('cpu')
