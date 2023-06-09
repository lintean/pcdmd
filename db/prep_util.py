#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   prep_util.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/6/9 17:06   lintean      1.0         None
'''
import copy
import os
from typing import List

import mne
import numpy as np
from mne.preprocessing import ICA, corrmap
from scipy import signal
from scipy.signal import hilbert, butter


def filter_voice(voice, l_freq, h_freq, original_fs=44100, fs=128, is_hilbert=True, is_p_law=True, internal_fs=8000, gl=150, gh=4000,
                 space=1.5, is_combine=True, *args, **kwargs):
    """
    语音数据预处理。包含希尔伯特变换、p-law变换和带通滤波（含降采样）
    :param voice: 原始的语音数据，列表形式，每个Trail为Time*2
    :param l_freq: 带通滤波的下限
    :param h_freq: 带通滤波的上限
    :param fs: 语音的输出频率
    :param is_hilbert: 是否进行希尔伯特变换
    :param is_p_law: 是否进行p-law变换
    """
    from eutils.audio import audspacebw, gammatonefir
    fs_voice = original_fs

    for k_tra in range(len(voice)):
        my_voice = voice[k_tra]
        my_voice = np.array(my_voice)

        # 降采样
        samples = int(my_voice.shape[1] / fs_voice * internal_fs)
        tmp_voice = [signal.resample(my_voice[0], samples), signal.resample(my_voice[1], samples)]
        my_voice = np.array(tmp_voice)

        # gammatone filter bank
        freqs = audspacebw(gl, gh, space)
        g = gammatonefir(freqs, fs=internal_fs, betamul=space)
        total_voice = []
        for sub in range(len(g[0])):
            sub_voice = signal.filtfilt(g[0][sub].squeeze(), [1], my_voice)

            # 希尔伯特变换
            if is_hilbert:
                sub_voice = hilbert(sub_voice)

            # p-law
            if is_p_law:
                sub_voice = abs(sub_voice)
                sub_voice = np.power(sub_voice, 0.6)

            # 降采样
            samples = int(sub_voice.shape[1] / internal_fs * fs)
            tmp_voice = [signal.resample(sub_voice[0], samples), signal.resample(sub_voice[1], samples)]
            sub_voice = np.array(tmp_voice)

            # 滤波
            wn = [l_freq / fs * 2, h_freq / fs * 2]
            # noinspection PyTupleAssignmentBalance
            b, a = butter(N=8, Wn=wn, btype='bandpass')
            sub_voice = signal.filtfilt(b, a, sub_voice)
            total_voice.append(sub_voice)

        # 合起来
        if is_combine:
            length = len(total_voice[0][0])
            my_voice = [np.zeros(length), np.zeros(length)]
            for i in range(len(total_voice)):
                my_voice[0] += total_voice[i][0]
                my_voice[1] += total_voice[i][1]
            my_voice[0] = my_voice[0][..., None]
            my_voice[1] = my_voice[1][..., None]
        else:
            my_voice = [[], []]
            for i in range(len(total_voice)):
                my_voice[0].append(total_voice[i][0])
                my_voice[1].append(total_voice[i][1])
            my_voice[0] = np.stack(my_voice[0], axis=-1)
            my_voice[1] = np.stack(my_voice[1], axis=-1)

        # 标准化
        my_voice[0] = (my_voice[0] - np.average(my_voice[0])) / np.std(my_voice[0])
        my_voice[1] = (my_voice[1] - np.average(my_voice[1])) / np.std(my_voice[1])
        voice[k_tra] = my_voice

    return voice


def ica_eeg(eeg, ica_dict, info, montage, eeg_tmp):
    """
    对数据进行ICA处理，去伪迹
    Args:
        eeg: 原始输入数据
        ica_dict: 手动选择的ICA成分
        info: 电极信息
        montage：电极布局信息
        eeg_tmp: 模板数据（S1-Trail1）

    Returns:
        data: 处理后的数据

    """

    eeg_tmp = eeg_tmp[0]
    eeg_tmp = np.transpose(eeg_tmp, (1, 0))

    # 计算ica通道
    raw_tmp = mne.io.RawArray(eeg_tmp[0:64, :], info)
    raw_tmp = raw_tmp.filter(l_freq=1, h_freq=None)
    raw_tmp.set_montage(montage)
    ica_tmp = ICA(n_components=20, max_iter='auto', random_state=97)
    ica_tmp.fit(raw_tmp)

    # 去眼电
    is_verbose = True

    for k_tra in range(len(eeg)):
        print(f'data ica, trail: {k_tra}')

        my_eeg = eeg[k_tra]
        my_eeg = np.transpose(my_eeg, [1, 0])

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(my_eeg[0:64, :], info, verbose=is_verbose)

        # 计算ica数据
        raw = raw.filter(l_freq=1, h_freq=None, verbose=is_verbose)
        ica = ICA(n_components=20, max_iter='auto', random_state=97, verbose=is_verbose)  # 97为随机种子
        ica.fit(raw)

        # 模板匹配法剔除眼电伪迹
        ica_exclude = []
        ica_s = [ica_tmp, ica]
        eog_channels = ica_dict  # 选取眼电通道
        for k_ica in range(len(eog_channels)):
            corrmap(ica_s, template=(0, eog_channels[k_ica]), threshold=0.9, label=str(k_ica), plot=False,
                    verbose=is_verbose)
            ica_exclude += ica_s[1].labels_[str(k_ica)]

        # 基于ICA去眼电
        ica.exclude = list(set(ica_exclude))
        ica.apply(raw, verbose=is_verbose)
        print(ica.exclude)
        del ica

        # 储存数据
        my_eeg = raw.get_data()
        my_eeg = np.transpose(my_eeg, [1, 0])
        eeg[k_tra] = my_eeg

        # 关闭可视化过程
        is_verbose = False

    return eeg


def filter_eeg(eeg, info, ref_channels='average', l_freq=1, h_freq=32, fs=128):
    """
    对数据进行滤波处理，并降低采样率到128Hz（标准化的采样率）
    Args:
        eeg: 去伪迹后的数据
        info: 电极信息
        ref_channels: 重参考的参考电极
        l_freq:带通滤波的低频范围
        h_freq:带通滤波的高频范围
        fs: 脑电的输出采样率
    Returns:
        data: 滤波后的数据

    """

    # 滤波
    is_verbose = True
    for k_tra in range(len(eeg)):
        print(f'data filter, trail: {k_tra}')

        my_eeg = eeg[k_tra]
        my_eeg = np.transpose(my_eeg, [1, 0])

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(my_eeg, info, verbose=is_verbose)

        # 重参考、滤波、降采样
        raw = raw.set_eeg_reference(ref_channels=ref_channels, verbose=is_verbose)
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=is_verbose)
        raw = raw.resample(fs)


        # 储存数据
        my_eeg = raw.get_data()[0:64, :]
        my_eeg = (my_eeg - np.average(my_eeg, axis=-1)[..., None]) / np.std(my_eeg, axis=-1)[..., None]
        my_eeg = np.transpose(my_eeg, [1, 0])
        eeg[k_tra] = my_eeg

        # 关闭可视化过程
        is_verbose = False

    return eeg


def voice_dirct2attd(voice, label):
    for i in range(len(voice)):
        if label[i]['direction'] == 1:
            voice[i][0], voice[i][1] = voice[i][1], voice[i][0]
    return voice


def voice_attd2speaker(voice, label):
    for i in range(len(voice)):
        if label[i]['speaker'] == 1:
            voice[i][0], voice[i][1] = voice[i][1], voice[i][0]
    return voice
