#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   functional.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/2/24 20:56   lintean      1.0         None
'''
import math

import numpy as np
import torch
from torch import nn


from typing import NamedTuple, Tuple, Optional
import torch.nn.init as init

from ..stbp.fire import threshold
from ..stbp.setting import SNNParameters, LIState, LIParameters


def lif_update(prev_v, prev_spike, current, vth, tau):
    v = tau * prev_v * (1 - prev_spike) + current
    spike = threshold(v - vth, "gauss", 0.0)
    return v, spike


def li_update(prev_v, current, tau):
    v = tau * prev_v + current
    return v


def opt_layer(opt, input_a, input_b, snn_process=True):
    if not snn_process:
        return opt(input_a, input_b)

    steps = input_a.shape[-1]
    output = []
    for step in range(steps):
        output.append(opt(input_a[..., step], input_b[..., step]))
    return torch.stack(output, dim=-1)


class ModuleLayer(nn.Module):
    """将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module): 需要转换的层。
            The layer needs to convert.
        bn (nn.Module): 如果需要加入BN，则将BN层一起当做参数传入。
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """

    def __init__(self, layer, bn=None, sp: SNNParameters = SNNParameters()):
        super(ModuleLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        self.sp = sp

    def forward(self, x):
        if not self.sp.snn_process:
            return self.layer(x.transpose(-1, -2)).transpose(-1, -2)

        steps = x.shape[-1]
        x_ = []
        for step in range(steps):
            x_.append(self.layer(x[..., step]))
        x_ = torch.stack(x_, dim=-1).contiguous()

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class Lift(nn.Module):
    def __init__(self, module: torch.nn.Module, sp: SNNParameters = SNNParameters()):
        super().__init__()
        self.lifted_module = module
        self.sp = sp

    def forward(self, x):
        if not self.sp.snn_process:
            return self.lifted_module(x)

        steps = x.shape[-1]
        x_ = []
        for step in range(steps):
            x_.append(self.lifted_module(x[..., step]))
        x_ = torch.stack(x_, dim=-1).contiguous()

        return x_


class TdBatchNorm3d(nn.BatchNorm3d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm3d
        eps (float): same with nn.BatchNorm3d
        momentum (float): same with nn.BatchNorm3d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm3d
        track_running_stats (bool): same with nn.BatchNorm3d
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True,
                 sp: SNNParameters = SNNParameters()):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = sp.vth_init
        self.snn_process = sp.snn_process

    def forward(self, x):
        '''
        @param x: shape (N, C, D, H, W)
        @return: shape (N, C, D, H, W)
        '''
        if not self.snn_process:
            return super().forward(x)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = x.mean([0, 2, 3, 4, 5])
            # use biased var in train
            var = x.var([0, 2, 3, 4, 5], unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = self.alpha * self.vth * (x - mean[None, :, None, None, None, None]) / (
            torch.sqrt(var[None, :, None, None, None, None] + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None, None, None, None] + self.bias[None, :, None, None, None, None]

        return x


class TdBatchNorm2d(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True,
                 sp: SNNParameters = SNNParameters()):
        super(TdBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = sp.vth_init
        self.snn_process = sp.snn_process

    def forward(self, x):
        '''
        @param x: shape (batch, channel, height, width, step)
        @return: shape (batch, channel, height, width, step)
        '''
        if not self.snn_process:
            return super().forward(x)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = x.mean([0, 2, 3, 4])
            # use biased var in train
            var = x.var([0, 2, 3, 4], unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = self.alpha * self.vth * (x - mean[None, :, None, None, None]) / (
            torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return x


class TdBatchNorm1d(nn.BatchNorm1d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm1d
        eps (float): same with nn.BatchNorm1d
        momentum (float): same with nn.BatchNorm1d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm1d
        track_running_stats (bool): same with nn.BatchNorm1d
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True,
                 sp: SNNParameters = SNNParameters(),):
        super(TdBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = sp.vth_init
        self.snn_process = sp.snn_process

    def forward(self, x):
        '''
        @param x: shape (batch, channel, step)
        @return: shape (batch, channel, step)
        '''
        if not self.snn_process:
            return super().forward(x)

        batch_size = x.shape[0]
        x = x.contiguous()
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            temp_x = x.view(batch_size, self.num_features, -1)
            mean = temp_x.mean([0, 2])
            var = temp_x.var([0, 2], unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = self.alpha * self.vth * (x - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x


def li_step(
    input_tensor: torch.Tensor,
    state: LIState,
    input_weights: torch.Tensor,
    bias: torch.Tensor,
    sp: SNNParameters = SNNParameters()
) -> Tuple[torch.Tensor, LIState]:
    r"""Single euler integration step of a leaky-integrator.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}


    and transition equations

    .. math::
        i = i + w i_{\text{in}}

    Parameters:
        input_tensor (torch.Tensor); Input spikes
        s (LIState): state of the leaky integrator
        input_weights (torch.Tensor): weights for incoming spikes
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """
    # compute current jumps
    i_new = torch.nn.functional.linear(input_tensor, input_weights)

    # compute voltage updates
    v_new = sp.tau_mem * state.v + i_new + bias

    return v_new, LIState(v_new, i_new)


class LILinearCell(torch.nn.Module):
    r"""Cell for a leaky-integrator with an additional linear weighting.
    More specifically it implements a discretized version of the ODE

    .. math::

        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}


    and transition equations

    .. math::
        i = i + w i_{\text{in}}

    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LIParameters): parameters of the leaky integrator
        dt (float): integration timestep to use
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LIParameters = LIParameters(),
        sp: SNNParameters = SNNParameters()
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sp = sp
        self.input_weights = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / np.sqrt(input_size)
        )
        self.bias = torch.nn.Parameter(
            torch.randn(self.hidden_size) / np.sqrt(hidden_size), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.input_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.input_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self, input_tensor: torch.Tensor, state: Optional[LIState] = None
    ) -> Tuple[torch.Tensor, LIState]:
        if state is None:
            state = LIState(
                v=self.sp.v_leak.detach(),
                i=torch.zeros(
                    (input_tensor.shape[0], self.hidden_size),
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                ),
            )
            state.v.requires_grad = True
        return li_step(
            input_tensor,
            state,
            self.input_weights,
            self.bias,
            sp=self.sp
        )


class ZeroExpandInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step):
        """
		Args:
			input: normalized within (0,1)
		"""
        st = torch.zeros(list(input.shape) + [step]).to(input.device)
        st[..., 0] = input
        return st

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None


class AllExpandInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step):
        """
		Args:
			input: normalized within (0,1)
		"""
        st = torch.zeros(list(input.shape) + [step]).to(input.device)
        for i in range(step):
            st[..., i] = input
        return st

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None