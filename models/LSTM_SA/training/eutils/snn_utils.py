#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   snn_utils.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/25 0:55   lintean      1.0         None
'''

import numpy as np

# 注意平方
def FLOPS_ann_conv(kernel_size, feature_size, c_in, c_out):
    return kernel_size * feature_size * c_in * c_out


def FLOPS_ann_fc(c_in, c_out):
    return c_in * c_out


def FLOPS_snn_conv(data, kernel_size, c_in):
    __time, __channel, __height, __width = data.shape
    all = __channel * __height * __width
    spike = np.sum(data)
    R = spike / all
    return R * FLOPS_ann_conv(kernel_size, __height * __width, c_in, __channel), spike, all


def FLOPS_snn_fc(data, c_in):
    __time, __c_out = data.shape
    all = __c_out
    spike = np.sum(data)
    R = spike / all
    return R * FLOPS_ann_fc(c_in, __c_out), spike, all


def energy_snn_conv(data, kernel_size, c_in):
    """
    计算snn中卷积层的energy 参考'Revisiting Batch Normalization for Training Low-latencyDeep Spiking Neural Networks from Scratch'
    @param c_in:
    @param kernel_size:
    @param data: 将要计算的data shape=[time, channel, height, width]
    @return: energy 或者 -1（计算失败）
    """
    snn_energy_coefficient = 0.9
    FLOPs, spike, all = FLOPS_snn_conv(data=data, kernel_size=kernel_size, c_in=c_in)
    return FLOPs * snn_energy_coefficient, spike, all


def energy_snn_fc(data, c_in):
    """
    计算snn中全连接层的energy 参考'Revisiting Batch Normalization for Training Low-latencyDeep Spiking Neural Networks from Scratch'
    @param c_in:
    @param data: 将要计算的data shape=[time, feature]
    @return: energy 或者 -1（计算失败）
    """
    snn_energy_coefficient = 0.9
    FLOPs, spike, all = FLOPS_snn_fc(data=data, c_in=c_in)
    return FLOPs * snn_energy_coefficient, spike, all


def energy_ann_conv(kernel_size, feature_size, c_in, c_out):
    """
    计算ann中卷积层的energy 参考'Revisiting Batch Normalization for Training Low-latencyDeep Spiking Neural Networks from Scratch'
    @param c_out:
    @param c_in:
    @param feature_size:
    @param kernel_size:
    @return: energy 或者 -1（计算失败）
    """
    ann_energy_coefficient = 4.6
    return FLOPS_ann_conv(kernel_size=kernel_size, feature_size=feature_size, c_in=c_in, c_out=c_out) * ann_energy_coefficient


def energy_ann_fc(c_in, c_out):
    """
    计算ann中全连接层的energy 参考'Revisiting Batch Normalization for Training Low-latencyDeep Spiking Neural Networks from Scratch'
    @param c_out:
    @param c_in:
    @return: energy 或者 -1（计算失败）
    """
    ann_energy_coefficient = 4.6
    return FLOPS_ann_fc(c_in=c_in, c_out=c_out) * ann_energy_coefficient