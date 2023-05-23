#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lsnn.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/8 19:57   lintean      1.0         None
'''
import math

import torch.nn as nn
import torch
from typing import NamedTuple, Tuple

from eutils.snn.stbp.fire import threshold
from eutils.snn.stbp.setting import SNNParameters


class LSNNState(NamedTuple):
    """State of an LSNN neuron

    Parameters:
        z (torch.Tensor): recurrent spikes 复发性尖峰
        v (torch.Tensor): membrane potential 膜电位
        i (torch.Tensor): synaptic input current 突触输入电流
        b (torch.Tensor): threshold adaptation 阈值适应
    """

    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor
    b: torch.Tensor


class LSNNParameters(NamedTuple):
    r"""Parameters of an LSNN neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\tau_\text{syn}`)
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\tau_\text{mem}`)
        tau_adapt_inv (torch.Tensor): adaptation time constant (:math:`\tau_b`)
        v_leak (torch.Tensor): leak potential
        v_th (torch.Tensor): threshold potential
        v_reset (torch.Tensor): reset potential
        beta (torch.Tensor): adaptation constant
    """

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    tau_adapt_inv: torch.Tensor = torch.as_tensor(1.0 / 800)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    beta: torch.Tensor = torch.as_tensor(1.8)
    method: str = "super"
    alpha: float = 100.0


def lsnn_step(
    input_tensor: torch.Tensor,
    state: LSNNState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    vth: torch.Tensor,
    p: LSNNParameters = LSNNParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LSNNState]:
    r"""Euler integration step for LIF Neuron with threshold adaptation
    More specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i \\
            \dot{b} &= -1/\tau_{b} b
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}} + b)

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}} \\
            b &= b + \beta z
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LSNNState): current state of the lsnn unit
        input_weights (torch.Tensor): synaptic weights for input spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LSNNParameters): parameters of the lsnn unit
        dt (float): Integration timestep to use
    """
    # compute voltage decay
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv

    # compute current decay
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute threshold adaptation update
    db = dt * p.tau_adapt_inv * (vth - state.b)
    b_decayed = state.b + db

    # compute new spikes
    z_new = threshold(v_decayed - b_decayed, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new.detach()) * v_decayed + z_new.detach() * p.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )

    b_new = b_decayed + z_new.detach() * p.beta
    return z_new, LSNNState(z_new, v_new, i_new, b_new)


class LSNNRecurrent(nn.Module):
    """A Long short-term memory neuron module *wit* recurrence
        adapted from https://arxiv.org/abs/1803.09574


    Parameters:
        input_size (int): Size of the input. Also known as the number of input features.
        hidden_size (int): Size of the hidden state. Also known as the number of input features.
        p (LSNNParameters): The neuron parameters as a torch Module, which allows the module
                to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: LSNNParameters = LSNNParameters(),
        sp: SNNParameters = SNNParameters(),
        autapses: bool = False,
        dt: float = 0.001,
    ):
        super().__init__()
        self.activation = lsnn_step
        self.autapses = autapses
        self.state_fallback = self.initial_state
        self.p = p
        self.sp = sp
        self.dt = dt
        self.input_size = torch.as_tensor(input_size)
        self.hidden_size = torch.as_tensor(hidden_size)

        self.input_weights = torch.nn.Parameter(
            torch.randn(self.hidden_size, self.input_size)
            * torch.sqrt(2.0 / self.hidden_size)
        )
        self.recurrent_weights = torch.nn.Parameter(
            torch.randn(self.hidden_size, self.hidden_size)
            * torch.sqrt(2.0 / self.hidden_size)
        )

        self.vth = sp.vth_init

        if not autapses:
            with torch.no_grad():
                self.recurrent_weights.fill_diagonal_(0.0)
            # Eradicate gradient updates from autapses
            def autapse_hook(gradient):
                return gradient.clone().fill_diagonal_(0.0)

            self.recurrent_weights.requires_grad = True
            self.recurrent_weights.register_hook(autapse_hook)

    def initial_state(self, input_tensor: torch.Tensor) -> LSNNState:
        dims = (*input_tensor.shape[1:-1], self.hidden_size)
        state = LSNNState(
            z=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=torch.full(
                dims,
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            b=torch.full(
                dims,
                torch.mean(self.p.v_th).detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.v.requires_grad = True
        return state

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}, p={self.p}, autapses={self.autapses}, dt={self.dt}"

    def forward(self, input_tensor, state=None):
        state = state if state is not None else self.state_fallback(input_tensor)

        T = input_tensor.shape[0]
        outputs = []

        for ts in range(T):
            out, state = self.activation(
                input_tensor[ts],
                state,
                self.input_weights,
                self.recurrent_weights,
                self.vth,
                self.p,
                self.dt,
            )
            outputs.append(out)

        return torch.stack(outputs), state
