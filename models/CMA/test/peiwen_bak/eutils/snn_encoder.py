#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   snn_encoder.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 14:05   lintean      1.0         None
'''

import torch.nn as nn
import torch
import math
from eutils.rate_encoders import BSAEncoder
import eutils.util as util


class BSA_Encoder(nn.Module):
    def __init__(self, args):
        super(BSA_Encoder, self).__init__()
        # 编码器
        self.args = args
        self.input_size = args.eeg_channel
        self.bsa_length = 7

        # 自己学习
        # bsa_weight = torch.empty(self.input_size, self.bsa_length)
        # nn.init.kaiming_uniform_(bsa_weight, a=math.sqrt(5))
        # bsa_weight -= torch.min(bsa_weight)

        # 使用前人文章的参数
        bsa_weight = torch.tensor([19, 74, 187, 240, 187, 74, 19], dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            bsa_weight = bsa_weight / 800
            bsa_weight = bsa_weight.expand((64, 7))
        self.bsa_weight = nn.Parameter(bsa_weight.clone())
        self.bsa = BSAEncoder(filter_response=self.bsa_weight, threshold=0.679, channel=args.eeg_channel)

    def encode(self, eeg):
        """
        对eeg数据进行bsa编码
        @param eeg: shape as [batch, channel, time]
        @return: encoded, origin(eeg after normalization), shapes as input
        """
        encoded, origin = [], []
        for i in range(self.args.batch_size):
            line = util.normalization(eeg[i, :, :], dim=1)
            origin.append(line)
            encoded.append(self.bsa.encode(line, filter_response=self.bsa_weight))
        encoded = torch.stack(encoded, dim=0)
        origin = torch.stack(origin, dim=0)
        return encoded, origin

    def decode(self, spike):
        """
        对spike数据进行bsa解码
        @param eeg: shape as [batch, channel, time]
        @return: encoded, origin(eeg after normalization), shapes as input
        """
        decoded = torch.nn.functional.conv1d(spike, torch.flip(self.bsa_weight, dims=[1]).unsqueeze(1),
                                             padding=self.bsa_length - 1, groups=self.input_size)[:, :,
                  :spike.shape[2]]

        return decoded


    def forward(self, x, targets):
        visualization_weights = []
        # visualization_weights.append(DotMap(data=data[0].transpose(1, 2).clone(), title="A", need_save=True,
        #                                     figsize=[data[0].shape[1] * 2, data[0].shape[2] * 2]))

        eeg = x[:, 0, self.args.audio_channel:-self.args.audio_channel, :]
        with torch.no_grad():
            encoded, origin = self.encode(eeg)
        decoded = self.decode(encoded)

        return decoded, origin, visualization_weights


