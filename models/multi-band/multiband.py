#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multiband.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/12 15:45   lintean      1.0         None
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torch.fft
from dotmap import DotMap
from eutils.torch.container import AADModel
from eutils.util import get_gpu_with_max_memory
import pytorch_warmup as warmup

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


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
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

    def forward(self, bs1, bs2, beeg, targets):
        visualization_weights = []

        wavA = bs1.transpose(-1, -2)
        eeg = beeg.transpose(-1, -2)
        wavB = bs2.transpose(-1, -2)

        # 时域数据转换成频域数据
        # wav and eeg shape: (Batch, Time, Channel)
        wavA = torch.abs(torch.fft.fft(wavA, dim=1))
        eeg = torch.abs(torch.fft.fft(eeg, dim=1))
        wavB = torch.abs(torch.fft.fft(wavB, dim=1))

        # 减少窗长
        wavA = wavA[:, :self.window_cp, :]
        eeg = eeg[:, :self.window_cp, :]
        wavB = wavB[:, :self.window_cp, :]
        self.args.window_length = self.window_cp

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
            temp1 = data[i * 3].view(self.args.batch_size, -1)
            data_dot = temp1 if data_dot is None else torch.cat([data_dot, temp1], dim=1)
        output = self.output_fc(data_dot)

        return output, targets, visualization_weights


def get_model(args: DotMap) -> AADModel:
    from ecfg import gpu_list
    model = Model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2 if "l2" in args else 0)
    scheduler = None
    warmup_scheduler = None
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=optimizer,
        sched=scheduler,
        warmup=warmup_scheduler,
        dev=torch.device(get_gpu_with_max_memory(gpu_list))
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model


