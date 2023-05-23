import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import eutils.util as util
from dotmap import DotMap
import torch.fft
import time

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_document_path = device_to_use.origin_data_document + "/KUL_single_single3"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "cnn"
args.ConType = ["No"]

args.CNN_file = "./CNN_normal.py"
args.CNN_split_file = "./CNN_split.py"
args.data_document = device_to_use.splited_data_document + "/???"

# 加载数据集元数据
data_meta = util.read_json(args.data_document_path + "/metadata.json")
args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
# args.names = ["S10"]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
args.window_length = math.ceil(data_meta.fs * 1)
# args.window_lap = math.ceil(data_meta.fs * 0.5)
args.window_lap = None
args.overlap = 1 - args.window_lap / args.window_length if args.window_lap is not None else 0
args.delay = 0
args.batch_size = 32
args.max_epoch = 100
args.min_epoch = 0
args.lr = 1e-3
args.early_patience = 0
args.random_seed = time.time()
args.cross_validation_fold = 5
args.current_flod = 0
args.one_hot_target = False

# 可视化选项 列表为空表示不希望可视化
args.visualization_epoch = []
args.visualization_window_index = []

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
args.test_percent = 0.2
args.vali_percent = 0

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
args.isDS = True
# DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
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

# 尝试启用gpu
# device = util.select_device(device_to_use.gpu_list)

# device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
device = torch.device('cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.hidden_channel = 10
        self.conv_eeg_audio_number = 1
        self.output_fc_number = self.conv_eeg_audio_number * self.hidden_channel
        self.length = 17
        self.left = 8
        self.right = 9

        self.conference_CNN = nn.Sequential(
            nn.Conv1d(args.eeg_channel, self.hidden_channel, self.length),
            nn.ELU(),
        )
        self.pool1d = nn.AdaptiveAvgPool1d(1)

        self.output_fc = nn.Sequential(
            # nn.BatchNorm1d(self.conv_output_channel),
            nn.Linear(self.output_fc_number, self.output_fc_number),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(self.output_fc_number, 2),
            nn.Sigmoid()
        )

        self.embedding_query = nn.Linear(self.hidden_channel, self.hidden_channel, bias=False)
        self.embedding_key = nn.Linear(self.hidden_channel, self.hidden_channel, bias=False)
        self.embedding_value = nn.Linear(self.hidden_channel, self.hidden_channel, bias=False)
        self.decoder_cma = nn.Linear(self.hidden_channel, self.hidden_channel, bias=False)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    # x shape: (Batch, Channel, Time)
    def update_qkv(self, q, k, v):
        q = self.embedding_query(q)
        k = self.embedding_key(k)
        v = self.embedding_value(v)

        return q, k, v

    def dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 去掉这个softmax：有效
        attn_weights = torch.bmm(q.transpose(1, 2), k)
        attn_weights = attn_weights / (args.eeg_channel ** 0.5)
        attn_weights = self.softmax(attn_weights)

        # 去掉+x：效果不明显
        attn_value = torch.bmm(attn_weights, v.transpose(1, 2))
        attn_value = attn_value.transpose(1, 2)
        # 仿照lif神经元
        attn_value = self.relu(attn_value)

        return attn_value, attn_weights

    # x shape: (Batch, Channel, Time)
    def cross_attention(self, q, k, v):
        q, k, v = self.update_qkv(q, k, v)
        attn_value, attn_weights = self.dot_product(q, k, v)
        attn_value = self.decoder_cma(attn_value)
        attn_value = self.relu(attn_value)
        return attn_value, attn_weights

    def truncated_cross_attention(self, q, k, v):
        q, k, v = self.update_qkv(q, k, v)
        step = q.shape[-1]
        attn_value = []
        attn_weights = []
        for i in range(step):
            qq = q[:, :, i: i + 1]
            start = i - self.left
            end = i + self.right
            kk = k[:, :, max(0, start): min(step, end)]
            vv = v[:, :, max(0, start): min(step, end)]
            value, weight = self.dot_product(qq, kk, vv)
            attn_value.append(value)
            pad = ((start < 0) * abs(start), (end >= step) * abs(end - step))
            attn_weights.append(nn.functional.pad(weight, pad))
        attn_value = torch.concat(attn_value, dim=-1)
        attn_weights = torch.concat(attn_weights, dim=1)
        attn_value = self.decoder_cma(attn_value)
        attn_value = self.relu(attn_value)
        return attn_value, attn_weights

    def forward(self, x, targets):
        visualization_weights = []

        # 读取输入
        # eeg = x[:, :, args.audio_channel:-args.audio_channel, :].squeeze()
        eeg_audio = x

        # CNN
        conv = nn.Unfold(kernel_size=(args.channel_number, self.length))
        eeg_audio = conv(eeg_audio)
        eeg_audio = eeg_audio.view(args.batch_size, args.channel_number, self.length, eeg_audio.shape[-1])
        # eeg_audio = eeg_audio.permute(0, 3, 1, 2)

        for i in range(eeg_audio.shape[-1]):
            audio1 = eeg_audio[:, :args.audio_channel, :, i]
            eeg = eeg_audio[:, args.audio_channel:-args.audio_channel, :, i]
            audio2 = eeg_audio[:, -args.audio_channel:, :, i]

            ae1 = self.cross_attention(audio1, eeg, eeg)
            ea1 = self.cross_attention(eeg, audio1, audio1)
            ae2 = self.cross_attention(audio2, eeg, eeg)
            ea2 = self.cross_attention(eeg, audio2, audio2)

        eeg = self.pool1d(eeg_audio)
        eeg = eeg.squeeze()
        output = self.output_fc(eeg)

        return output, targets, visualization_weights


# 模型参数和初始化
myNet = CNN()
clip = 0.8
optimzer = torch.optim.Adam(myNet.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimzer, mode='min', factor=0.5, patience=300, verbose=True,
                                                       threshold=1e-5, threshold_mode='rel', cooldown=5, min_lr=0,
                                                       eps=1e-8)
loss_func = nn.CrossEntropyLoss()
