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
args.data_document_path = device_to_use.origin_data_document + "/KUL_multiple_multiple3"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "KUL_Linear_final"
args.ConType = ["No"]

args.CNN_file = "./CNN_normal.py"
args.CNN_split_file = "./CNN_split.py"
args.data_document = device_to_use.splited_data_document + "/KUL_multiple_multiple6"

# 加载数据集元数据
data_meta = util.read_json(args.data_document_path + "/metadata.json")
args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
# args.names = ["S12", "S13"]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
args.window_length = math.ceil(data_meta.fs * 0.5)
args.window_lap = math.ceil(data_meta.fs * 0.25)
# args.window_lap = None
args.overlap = 1 - args.window_lap / args.window_length if args.window_lap is not None else 0.8
args.delay = 0
args.batch_size = 16
args.max_epoch = 100
args.min_epoch = 0
args.early_patience = 0
args.random_seed = time.time()
args.cross_validation_fold = 5
# args.current_flod = 1

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
args.test_percent = 0.1
args.vali_percent = 0.1

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
args.isDS = True
# DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
args.isFM = 1
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

from torch.nn import Parameter

# Code adapted from the fairseq repo.
class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, scale_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.scale_dim = scale_dim
        self.head_dim = self.scale_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * self.scale_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.scale_dim))

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.scale_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.scale_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)


    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()

        # 编码
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 相乘
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / ((k.shape[1] * q.shape[1]) ** 0.25)

        return None, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.scale_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.scale_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.scale_dim, end=2 * self.scale_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.scale_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_output_channel = 10
        self.conv_eeg_audio_number = 4
        self.output_fc_number = self.conv_eeg_audio_number * self.conv_output_channel

        self.channel = [10, 14, 14, 10]
        self.q_size = [10, 320, 320, 10]
        self.kv_size = [320, 10, 10, 320]
        self.ofc_channel = args.audio_band * args.eeg_band

        self.output_fc = nn.Sequential(
            nn.Linear(self.ofc_channel * 2, self.ofc_channel), nn.ReLU(),
            nn.Linear(self.ofc_channel, 2), nn.Sigmoid()
        )

        self.window_cp = math.floor(args.window_length / 2)

        self.channel_num = [args.audio_band, args.eeg_band, args.audio_band]
        self.channel_origin = [args.audio_channel_per_band, args.eeg_channel_per_band, args.audio_channel_per_band]
        self.channel_target = [1, 1, 1]
        self.eeg_change = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.channel_origin[index], self.channel_origin[index]),
                    nn.GELU(),
                    nn.Linear(self.channel_origin[index], self.channel_target[index])
                )
                for i in range(self.channel_num[index])])
            for index in range(3)])

        self.cm_attn = nn.ModuleList([MultiheadAttention(
            embed_dim=self.window_cp,
            scale_dim=32,
            num_heads=1,
            attn_dropout=0
        ) for i in range(self.conv_eeg_audio_number)])

    # data shape: (Batch, Channel * bands, Time)
    def split_bands(self, data, band_number):
        temp = []
        channel = int(data.shape[2] / band_number)
        for i in range(band_number):
            temp.append(data[:, :, i * channel:(i + 1) * channel])
        return temp

    def forward(self, x):
        visualization_weights = []

        wavA = x[:, 0, 0:args.audio_channel, :]
        wavA = wavA.transpose(1, 2)
        eeg = x[:, 0, args.audio_channel:-args.audio_channel, :]
        eeg = eeg.transpose(1, 2)
        wavB = x[:, 0, -args.audio_channel:, :]
        wavB = wavB.transpose(1, 2)

        # 时域数据转换成频域数据
        # wav and eeg shape: (Batch, Time, Channel)
        wavA = torch.abs(torch.fft.fft(wavA, dim=1))
        eeg = torch.abs(torch.fft.fft(eeg, dim=1))
        wavB = torch.abs(torch.fft.fft(wavB, dim=1))

        # 减少窗长
        wavA = wavA[:, :self.window_cp, :]
        eeg = eeg[:, :self.window_cp, :]
        wavB = wavB[:, :self.window_cp, :]
        args.window_length = self.window_cp

        # 通道对齐
        split = [wavA, eeg, wavB]
        # wav and eeg shape: (Batch, Time, Channel)
        for index in range(len(split)):
            split[index] = self.split_bands(split[index], self.channel_num[index])
            for i in range(len(split[index])):
                split[index][i] = self.eeg_change[index][i](split[index][i])
            split[index] = torch.cat(split[index], dim=2)
        wavA, eeg, wavB = split

        # 4CMA
        # multihead_attention Input shape: Time x Batch x Channel
        # wav and eeg shape: (Batch, Channel, Time)
        data = [wavA, eeg, eeg, wavB]
        kv = [eeg, wavA, wavB, eeg]
        weight = [0 for i in range(len(data))]
        # visualization_weights.append(DotMap(data=data[0][0].clone(), title="bef_q"))
        # visualization_weights.append(DotMap(data=kv[0][0].clone(), title="bef_kv"))
        for j in range(2):
            i = j * 3
            data[i] = data[i].permute(2, 0, 1)
            kv[i] = kv[i].permute(2, 0, 1)
            weight[i], data[i] = self.cm_attn[i](data[i], kv[i], kv[i])
            # data[i] = data[i].permute(1, 2, 0)
            # 频域数据转换成时域数据
            # data[i] = torch.abs(torch.fft.ifft(data[i], dim=1))
            # time和channel相互替代
            # data[i] = data[i].transpose(1, 2)
        # visualization_weights.append(DotMap(data=data[0].transpose(1, 2)[0].clone(), title="aft"))
        # visualization_weights.append(DotMap(data=weight[0].clone(), title="eeg_v", need_save=True))
        # visualization_weights.append(DotMap(data=weight[1].clone(), title="audio_v", need_save=True))

        # dot
        # wav and eeg shape: (Batch, Channel, Time)
        data_dot = None
        for i in range(2):
            temp1 = data[i * 3].view(args.batch_size, -1)
            data_dot = temp1 if data_dot is None else torch.cat([data_dot, temp1], dim=1)
        output = self.output_fc(data_dot)

        return output, visualization_weights


# 模型参数和初始化
myNet = CNN()
clip = 0.8

lr = 2e-4
optimzer = torch.optim.Adam(myNet.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
# device = torch.device('cpu')