#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   snn_utils.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/25 0:55   lintean      1.0         None
'''

import numpy as np
from .container import SPower
from thop import profile

ann_energy_coefficient = 4.6
snn_energy_coefficient = 0.9


def mac_ann_conv(kernel_size, feature_size, c_in, c_out):
    return kernel_size * feature_size * c_in * c_out


def mac_ann_fc(c_in, c_out):
    return c_in * c_out


def get_firing_rate(data, return_spower=False):
    if return_spower:
        return SPower(max_spikes=np.product(data.shape), spikes=np.count_nonzero(data))
    else:
        return np.count_nonzero(data) / np.product(data.shape)


def get_snn_energy(ann_power: SPower, firing_rate: SPower):
    rate = firing_rate.spikes / firing_rate.max_spikes
    return SPower(
        energy=ann_power.mac * rate * snn_energy_coefficient,
        mac=ann_power.mac * rate,
        spikes=firing_rate.spikes,
        max_spikes=firing_rate.max_spikes,
        params=ann_power.params
    )


def energy_ann_conv(kernel_size, feature_size, c_in, c_out):
    """
    计算ann中卷积层的energy 参考'Revisiting Batch Normalization for Training Low-latencyDeep Spiking Neural Networks from Scratch'
    @param batch_size:
    @param c_out:
    @param c_in:
    @param feature_size:
    @param kernel_size:
    @return: energy 或者 -1（计算失败）
    """
    macs = mac_ann_conv(
            kernel_size=kernel_size,
            feature_size=feature_size,
            c_in=c_in,
            c_out=c_out
    )

    return SPower(
        energy=macs * ann_energy_coefficient,
        mac=macs
    )


def energy_ann_fc(c_in, c_out) -> SPower:
    """
    计算ann中全连接层的energy 参考'Revisiting Batch Normalization for Training Low-latencyDeep Spiking Neural Networks from Scratch'
    @param c_out:
    @param c_in:
    @return: energy 或者 -1（计算失败）
    """
    macs = mac_ann_fc(c_in=c_in, c_out=c_out)
    return SPower(
        energy=macs * ann_energy_coefficient,
        mac=macs
    )


def ann_energy(module, data_in, *args, **kwargs) -> SPower:
    """
    计算ANN的MAC数量 注意卷积的bias
    @param module:
    @param data_in:
    @return:
    """
    macs, params = profile(module, inputs=(data_in,), *args, **kwargs)
    return SPower(
        energy=macs * ann_energy_coefficient,
        mac=macs,
        params=params
    )


def snn_energy(module, data_in, data_out, *args, **kwargs) -> SPower:
    """
    计算SNN的MAC数量
    @param data_out:
    @param module:
    @param data_in:
    @return:
    """
    macs, params = profile(module, inputs=(data_in,), *args, **kwargs)
    return SPower(
        energy=macs * (np.count_nonzero(data_out) / np.product(data_out.shape)) * snn_energy_coefficient,
        mac=macs * (np.count_nonzero(data_out) / np.product(data_out.shape)),
        spikes=np.count_nonzero(data_out),
        max_spikes=np.product(data_out.shape),
        params=params
    )


def energy(snn_process=False, *args, **kwargs) -> SPower:
    """
    计算ANN或SNN的MAC数量
    @param snn: 是否是snn
    @param module:
    @param data:
    @return:
    """
    if snn_process:
        return snn_energy(*args, **kwargs)
    else:
        return ann_energy(*args, **kwargs)