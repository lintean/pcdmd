import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from ecfg import *
import eutils.util as util
from dotmap import DotMap
import torch.fft

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_document_path = origin_data_document + "/KUL_band20210124"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "mb_cnn_se"
args.ConType = ["No"]

args.CNN_file = "./CNN_normal.py"
args.CNN_split_file = "./CNN_split.py"
args.data_document = "./data/mb_kul_try_normal"

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

from torch.nn import Parameter


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
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
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / k.shape[1] ** 0.5

        # print(attn_weights.shape)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights = attn_weights + attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

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

        self.channel = [10, 320, 10]
        self.net_eeg_channel_per_band = 1
        self.ofc_channel = self.net_eeg_channel_per_band * args.eeg_band * args.audio_channel
        self.conv_output_channel = 10
        self.conv_eeg_audio_number = 4
        self.output_fc_number = self.conv_output_channel * self.conv_eeg_audio_number

        self.conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.channel[i], self.conv_output_channel, 9),
            nn.ReLU(),
        ) for i in range(len(self.channel))])

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.conference_CNN = nn.Sequential(
            nn.Conv1d(64 * 4, self.conv_output_channel, 9),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.output_fc = nn.Sequential(
            nn.Linear(30, 30), nn.ReLU(),
            nn.Linear(30, 2), nn.Sigmoid()
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

        self.eeg_change = nn.ModuleList([
            nn.Conv1d(args.eeg_channel_per_band, self.net_eeg_channel_per_band, 1, padding=0, bias=False
        ) for i in range(args.eeg_band)])
        self.proj_images = nn.Conv1d(args.eeg_channel, args.audio_channel, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(args.eeg_channel, args.audio_channel, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(args.audio_channel, args.eeg_channel, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(args.audio_channel, args.eeg_channel, 1, bias=False)

        # self.cm_attn = nn.ModuleList([MultiheadAttention(
        #     embed_dim=16,
        #     num_heads=1,
        #     attn_dropout=0
        # ) for i in range(conv_eeg_audio_number)])

        q_size = [10, 320, 320, 10]
        kv_size = [320, 10, 10, 320]
        self.cm_attn = nn.ModuleList([MultiheadAttention(
            embed_dim=256,
            num_heads=1,
            # layers=5,
            # attn_dropout=0,
            # relu_dropout=0,
            # res_dropout=0,
            # embed_dropout=0,
            # attn_mask=False
        ) for i in range(self.conv_eeg_audio_number)])

        self.lstm_wavA = nn.LSTM(16, 16, 8)
        self.lstm_wavB = nn.LSTM(16, 16, 8)
        self.lstm_eeg = nn.LSTM(16, 16, 8)


    def dot(self, a, b):
        # 皮尔逊相似度
        # mean_a = torch.mean(a, dim=0)
        # mean_b = torch.mean(b, dim=0)
        # a = a - mean_a
        # b = b - mean_b
        # 通道余弦
        temp = torch.tensordot(a, b, dims=[[0], [0]])
        norm_a = torch.norm(a, p=2, dim=0).unsqueeze(1)
        norm_b = torch.norm(b, p=2, dim=0).unsqueeze(0)
        norm_ab = torch.mm(norm_a, norm_b)
        temp = torch.div(temp, norm_ab)
        # 通道相干性
        # temp = torch.zeros(a.shape[1], b.shape[1]).to(device)
        # for i in range(temp.shape[0]):
        #     for j in range(temp.shape[1]):
        #         temp[i, j] = torch.mean(self.coherence(dps_seq, a[:, i], b[:, j]))
        # 压平后输出
        return temp

    def Euclidean_Distance(self, a, b):
        # 欧氏距离
        temp = a - b
        temp = torch.norm(temp, p=2, dim=0)
        return temp

    # data shape: (Batch, Channel * bands, Time)
    def split_bands(self, data):
        temp = []
        channel = int(data.shape[1] / args.eeg_band)
        for i in range(args.eeg_band):
            temp.append(data[:, i*channel:(i+1)*channel, :])
        return temp

    # data shape: (Batch, Channel * bands, Time)
    # output shape: (Batch, Channel * Time, bands)
    def combine_bands(self, data):
        data = self.split_bands(data)
        for i in range(args.eeg_band):
            data[i] = data[i].view(1, -1, 1)

        temp = torch.cat([data[i] for i in range(len(data))], dim=2)
        return temp

    def spectral_density(self, dps_seq, x, y):
        xy = None
        for i in range(dps_seq.shape[0]):
            temp_x = x * dps_seq[i, :]
            temp_y = y * dps_seq[i, :]
            temp_x = torch.fft.fft(temp_x)
            temp_y = torch.fft.fft(temp_y)
            xy = temp_x * temp_y if xy is None else xy + temp_x * temp_y

        return torch.view_as_real(xy / dps_seq.shape[0])

    def coherence(self, dps_seq, x, y):
        up = self.spectral_density(dps_seq, x, y)
        down = torch.sqrt(self.spectral_density(dps_seq, x, x) * self.spectral_density(dps_seq, y, y))
        return torch.flatten(up / down)

    def forward(self, x):
        wavA = x[0, 0, 0:args.audio_channel, :]
        wavA = torch.t(wavA).unsqueeze(0)
        eeg = x[0, 0, args.audio_channel:-args.audio_channel, :]
        eeg = torch.t(eeg).unsqueeze(0)
        wavB = x[0, 0, -args.audio_channel:, :]
        wavB = torch.t(wavB).unsqueeze(0)

        origin_wava = wavA.clone()
        origin_eeg = eeg.clone()
        origin_wavb = wavB.clone()

        # 时域数据转换成频域数据
        # wavA = torch.abs(torch.fft.fft(wavA, dim=1))
        # eeg = torch.abs(torch.fft.fft(eeg, dim=1))
        # wavB = torch.abs(torch.fft.fft(wavB, dim=1))

        wavA = wavA.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        wavB = wavB.transpose(1, 2)

        eeg = self.split_bands(eeg)
        for i in range(len(eeg)):
            eeg[i] = self.eeg_change[i](eeg[i])
        eeg = torch.cat(eeg, dim=1)

        # wav Channel 1 to 16
        # wavA = self.proj_audio(wavA)
        # wavB = self.proj_audio2(wavB)
        # eeg = self.proj_images(eeg)

        wavA = wavA.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        wavB = wavB.transpose(1, 2)

        # change the bands
        # wavA = self.combine_bands(wavA)
        # eeg = self.combine_bands(eeg)
        # wavB = self.combine_bands(wavB)

        # 4CMA
        # multihead_attention Input shape: Time x Batch x Channel
        # wav and eeg shape: (Batch, Channel, Time)
        data = [wavA, eeg, eeg, wavB]
        kv = [eeg, wavA, wavB, eeg]
        weight = [0 for i in range(self.conv_eeg_audio_number)]
        for i in range(self.conv_eeg_audio_number):
            data[i] = data[i].permute(2, 0, 1)
            kv[i] = kv[i].permute(2, 0, 1)
            data[i], weight[i] = self.cm_attn[i](data[i], kv[i], kv[i])
            data[i] = data[i].permute(1, 2, 0)
            # time和channel相互替代
            # data[i] = data[i].transpose(1, 2)
            # 频域数据转换成时域数据
            # data[i] = torch.abs(torch.fft.ifft(data[i], dim=1))

        # dot
        # wav and eeg shape: (Batch, Channel, Time)
        data_dot = []
        for i in range(2):
            temp1 = self.dot(data[i * 3].squeeze(0), data[i + 1].squeeze(0))
            # temp2 = self.Euclidean_Distance(data[i * 3].squeeze(0), data[i + 1].squeeze(0))
            # temp1 = self.fc[i](temp1.unsqueeze(1))
            # temp2 = self.fc2[i](temp2.unsqueeze(1))
            # data_dot = temp1.squeeze(1) if data_dot is None else torch.cat([data_dot, temp1.squeeze(1)], dim=0)
            # data_dot = torch.cat([data_dot, temp2.squeeze(1)], dim=0)
            data_dot.append(temp1)
        all_weight = torch.abs(data_dot[0] - data_dot[1])

        wav_weight = torch.mean(all_weight, dim=1)
        eeg_weight = torch.mean(all_weight, dim=0)

        # 加权
        origin_wava = origin_wava.transpose(1,2)
        origin_eeg  = origin_eeg.transpose(1, 2)
        origin_wavb = origin_wavb.transpose(1, 2)

        origin_eeg = self.split_bands(origin_eeg)
        weighted_wava = []
        weighted_wavb = []
        for i in range(wav_weight.shape[0]):
            weighted_wava.append(origin_wava[:, i, :] * wav_weight[i])
            weighted_wavb.append(origin_wavb[:, i, :] * wav_weight[i])
        weighted_wava = torch.stack(weighted_wava, dim=1)
        weighted_wavb = torch.stack(weighted_wavb, dim=1)

        for i in range(eeg_weight.shape[0]):
            origin_eeg[i] = origin_eeg[i] * eeg_weight[i]
        weighted_egg = torch.cat(origin_eeg, dim=1)

        # CNN
        data_cnn = [weighted_wava, weighted_egg, weighted_wavb]
        for i in range(3):
            data_cnn[i] = self.conv[i](data_cnn[i])
            data_cnn[i] = self.pool(data_cnn[i]).view(-1, 10)

        x = torch.cat([data_cnn[i] for i in range(3)], dim=1)
        output = self.output_fc(x)

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
optimzer = torch.optim.Adam(myNet.parameters(), lr=lr)
# optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.choice(gpu_list)
device = torch.device('cuda:' + str(gpu_random))
# device = torch.device('cpu')