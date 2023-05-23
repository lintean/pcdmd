import torch
import torch.nn as nn
import torch.nn.functional as F
from ecfg import *
# from eutils.util import heatmap

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "4CMA+CNN_默认初始化"
ConType = ["No"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
# names = ["S2"]

# 所用的数据目录路径
data_document_path = origin_data_document + "/AAD_KUL2"

CNN_file = "./CNN_normal.py"
CNN_split_file = "./CNN_split.py"
data_document = "./data/split_1_new"

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
people_number = 16
eeg_channel = 64
audio_channel = 1
channel_number = eeg_channel + audio_channel * 2
trail_number = 8
cell_number = 25200
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

# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

# 整体模型
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

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.out_proj = nn.LSTM(16, 16, 8)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
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
            nn.Linear(output_fc_number, output_fc_number), nn.ReLU(),
            nn.Linear(output_fc_number, 2), nn.Sigmoid()
        )

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])
        self.average = nn.AdaptiveAvgPool1d(1)

        self.proj_images = nn.Conv1d(eeg_channel, 16, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(eeg_channel, 16, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(audio_channel, 16, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(audio_channel, 16, 1, bias=False)

        self.cm_attn = nn.ModuleList([MultiheadAttention(
            embed_dim=16,
            num_heads=1,
            attn_dropout=0
        ) for i in range(conv_eeg_audio_number)])

        self.lstm = nn.LSTM(16, 16, 8)

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

        # print(wavA.shape)
        # print(eeg.shape)

        # wav Channel 1 to 16
        # wavA.register_hook(save_grad('1convA'))
        # wavB.register_hook(save_grad('1convB'))
        wavA = self.proj_audio(wavA)
        wavB = self.proj_audio2(wavB)
        eeg = self.proj_images(eeg)


        # print(wavA.shape)
        # print(eeg.shape)

        # heatmap(eeg[0])

        # 4CMA
        # multihead_attention Input shape: Time x Batch x Channel
        # wav and eeg shape: (Batch, Channel, Time)
        data = [wavA, eeg, eeg, wavB]
        kv = [eeg, wavA, wavB, eeg]
        weight = [0 for i in range(conv_eeg_audio_number)]
        for i in range(conv_eeg_audio_number):
            # data[i].register_hook(save_grad('CMA' + str(i)))
            data[i] = data[i].permute(2, 0, 1)
            kv[i] = kv[i].permute(2, 0, 1)
            data[i], weight[i] = self.cm_attn[i](query=data[i], key=kv[i], value=kv[i])
            data[i] = data[i].permute(1, 2, 0)
            # data[i].register_hook(save_grad('conv' + str(i)))

        # heatmap(data[0][0])

        # # LSTM input shape: Time x Batch x Channel
        # wavA = wavA.permute(2, 0, 1)
        # wavB = wavB.permute(2, 0, 1)
        # wavA, _ = self.lstm(wavA)
        # wavB, _ = self.lstm(wavB)
        # wavA = wavA.permute(1, 2, 0)
        # wavB = wavB.permute(1, 2, 0)

        # dot
        # wav and eeg shape: (Batch, Channel, Time)
        # data_dot = None
        # for i in range(2):
        #     temp = torch.tensordot(data[i * 3].squeeze(0), data[i + 1].squeeze(0), dims=[[0], [0]])
        #     temp = torch.diag(temp)
        #     temp = self.fc[i](temp.unsqueeze(1))
        #     data_dot = temp.squeeze(1) if data_dot is None else torch.cat([data_dot, temp.squeeze(1)], dim=0)
        # output = self.output_fc(data_dot).unsqueeze(0)

        # CNN
        for i in range(conv_eeg_audio_number):
            data[i] = self.conv[i](data[i])
            # if i == 0:
                # data[0].register_hook(save_grad('pool0'))
            data[i] = self.pool(data[i]).view(-1, conv_output_channel)

        x = torch.cat([data[i] for i in range(conv_eeg_audio_number)], dim=1)
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