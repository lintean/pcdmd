import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import NamedTuple, Tuple, Optional
from eutils.util import normalization


class SNNParameters(NamedTuple):
    snn_process: bool = True
    vth_dynamic: bool = True
    vth_init: float = 0.5
    vth_low: float = 0.1
    vth_high: float = 0.9
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    vth_random_init: bool = True
    tau_mem: float = 0.25
    tau_syn: float = 0.25
    tau_adapt: float = 0.25
    method: str = "gauss"


@torch.jit.script
def heaviside(data):
    r"""
    A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
    that truncates numbers <= 0 to 0 and everything else to 1.

    .. math::
        H[n]=\begin{cases} 0, & n <= 0 \\ 1, & n \gt 0 \end{cases}
    """
    return torch.gt(data, torch.as_tensor(0.0)).to(data.dtype)  # pragma: no cover


class SuperSpike(torch.autograd.Function):
    r"""SuperSpike surrogate gradient as described in Section 3.3.2 of
    一种模拟偏导数方法，使用快速sigmoid f(x)=x/(1 + |x|)的负半部分的偏导数

    F. Zenke, S. Ganguli, **"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"**,
    Neural Computation 30, 1514–1541 (2018),
    `doi:10.1162/neco_a_01086 <https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086>`_
    """

    @staticmethod
    @torch.jit.ignore
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return heaviside(input_tensor)

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(2)  # section 3.3.2 (beta -> alpha)
        return grad, None


class SpikeAct(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return heaviside(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        # hu = abs(input) < aa
        # hu = hu.float() / (2 * aa)
        hu = (1 / np.sqrt(0.3 * np.pi)) * torch.exp(-torch.pow(input, 2) / 0.3)
        return grad_input * hu


def threshold(x: torch.Tensor, method: str, alpha: float = 100.0) -> torch.Tensor:
    """
    x < 0 return 0 or x >= 0 return 1, 等价于 x >= 0
    @param x:
    @param method:
    @param alpha:
    @return:
    """
    if method == "gauss":
        return SpikeAct.apply(x)
    elif method == "super":
        return SuperSpike.apply(x, torch.as_tensor(alpha))


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n, vth, tau):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = threshold(u_t1_n1 - vth, "gauss", 0.0)
    return u_t1_n1, o_t1_n1


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

    def __init__(self, layer, bn = None, sp: SNNParameters = SNNParameters()):
        super(ModuleLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        self.sp = sp

    def forward(self, x):
        if not self.sp.snn_process:
            return self.layer(x)

        steps = x.shape[-1]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class LIFSpike(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, activation = None, sp: SNNParameters = SNNParameters()):
        super(LIFSpike, self).__init__()
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
            u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step], self.vth, self.sp.tau_mem)
        return out


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
                 snn_process=True, vth=0.2):
        super(TdBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = vth
        self.snn_process = snn_process

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
                 snn_process=True, sp: SNNParameters = SNNParameters(),):
        super(TdBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = sp.vth_init
        self.snn_process = snn_process

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


class LIParameters(NamedTuple):
    """Parameters of a leaky integrator

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time constant
        tau_mem_inv (torch.Tensor): inverse membrane time constant
        v_leak (torch.Tensor): leak potential
    """

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)


class LIState(NamedTuple):
    """State of a leaky-integrator

    Parameters:
        v (torch.Tensor): membrane voltage
        i (torch.Tensor): input current
    """

    v: torch.Tensor
    i: torch.Tensor


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
