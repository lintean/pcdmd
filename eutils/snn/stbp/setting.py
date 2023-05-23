#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   setting.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/2/24 20:30   lintean      1.0         None
'''
import torch
from typing import NamedTuple, Tuple, Optional


class SNNParameters(NamedTuple):
    snn_process: bool = True
    vth_dynamic: bool = True
    vth_init: float = 1
    vth_low: float = 0.1
    vth_high: float = 0.9
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    vth_random_init: bool = True
    tau_mem: float = 0.25
    tau_syn: float = 0.2
    tau_adapt: float = 0.25
    method: str = "gauss"


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