#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   neurons.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/2/24 20:28   lintean      1.0         None
'''
import torch
import torch.nn as nn

from ..stbp.functional import lif_update, li_update
from ..stbp.setting import SNNParameters


class LIF(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, activation=None, sp: SNNParameters = SNNParameters()):
        super().__init__()
        self.sp = sp
        self.activation = activation
        self.vth = self.sp.vth_init

    def forward(self, x):
        if not self.sp.snn_process:
            return self.activation(x)

        steps = x.shape[-1]
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = lif_update(u, out[..., max(step - 1, 0)], x[..., step], self.vth, self.sp.tau_mem)
        return out


class LI(nn.Module):
    def __init__(self, activation=None, sp: SNNParameters = SNNParameters()):
        super().__init__()
        self.sp = sp
        self.activation = activation
        self.vth = self.sp.vth_init

    def forward(self, x):
        if not self.sp.snn_process:
            return self.activation(x)

        steps = x.shape[-1]
        u_ = []
        u = torch.zeros(x.shape[:-1], device=x.device)
        for step in range(steps):
            u = li_update(u, x[..., step], self.sp.tau_mem)
            u_.append(u)
        u_ = torch.stack(u_, dim=-1)
        return u_