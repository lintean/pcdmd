#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fire.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/2/24 20:32   lintean      1.0         None
'''
import numpy as np
import torch


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
