#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/24 10:00   lintean      1.0         None
'''

# 读取数据，包含脑电信号、音频信号和标签信息
import copy
import os
from typing import List

import mne
import numpy as np
import scipy.io
from mne.preprocessing import ICA, corrmap
from scipy import signal
from scipy.io import wavfile
from scipy.signal import hilbert, butter
from .DTU import dtu_eeg_fs, dtu_audio_fs, dtu_label, trail_number, channel_names_scut, DTUMeta, name
from .util import *

dataset_name = name
path = "../AAD_SCUT"
montage = 'standard_1020'
fs = dtu_eeg_fs


def preprocess(dataset_path="../AAD_SCUT", sub_id="1", eeg_lf: int or List[int] = 1, eeg_hf: int or List[int] = 32,
                wav_lf: int or List[int] = 1, wav_hf: int or List[int] = 32,
               ica=True, label_type='speaker', need_voice=True, *args, **kwargs):
    """
    读取数据库的数据，经过ICA去伪迹，然后带通滤波，最后输出标准的样本及标签
    Args:
        dataset_path: 数据库路径
        sub_id: 需要提取的受试者编号
        l_freq: 带通滤波器参数，低频下限
        h_freq: 带通滤波器参数，高通上限
        is_ica: 是否进行ICA处理

    Returns:
        eeg：list，each item is a trail: np.array[Time*Channels]
        voice：list，each item is a trail: list[np.array(Time*Channels)], the first one is attend
        label：list，each item is a trail: list[方位、讲话者]，以0、1标记

    """
    global path
    path = dataset_path

    if isinstance(eeg_lf, int) and isinstance(eeg_hf, int):
        eeg_lf, eeg_hf = [eeg_lf], [eeg_hf]
    if isinstance(wav_lf, int) and isinstance(wav_hf, int):
        wav_lf, wav_hf = [wav_lf], [wav_hf]

    # 加载数据
    eeg, voice, label = data_loader(sub_id)
    meta = DTUMeta

    # 分别处理脑电、语音、label
    eegs, voices = [], []
    if need_voice:
        for idx in range(len(wav_lf)):
            voices.append(preprocess_voice(voice.copy(), label, wav_lf[idx], wav_hf[idx], label_type, *args, **kwargs))

        meta["wav_band"] = voices[0][0][0].shape[-1]
        meta["wav_band_chan"] = 1
        voice_data = []
        for trail_idx in range(len(voices[0])):
            trail_data = []
            for audio_idx in range(len(voices[0][trail_idx])):
                audio_data = []
                for idx in range(len(voices)):
                    audio_data.append(voices[idx][trail_idx][audio_idx])
                audio_data = np.concatenate(audio_data, axis=-1)
                trail_data.append(audio_data)
            voice_data.append(trail_data)
        voices = voice_data
    else:
        voices = None
    voice = voices

    for idx in range(len(eeg_lf)):
        eegs.append(preprocess_eeg(eeg.copy(), eeg_lf[idx], eeg_hf[idx], ica, *args, **kwargs))
    meta["eeg_band"] = len(eeg_lf)
    meta["eeg_band_chan"] = eegs[0][0].shape[-1]
    eeg_data = []
    for trail_idx in range(len(eegs[0])):
        trail_data = []
        for idx in range(len(eegs)):
            trail_data.append(eegs[idx][trail_idx])
        trail_data = np.concatenate(trail_data, axis=-1)
        eeg_data.append(trail_data)
    eeg = eeg_data
    label = select_label(label, label_type)

    meta["subj_id"] = sub_id
    meta["eeg_fs"] = 128
    meta["wav_fs"] = 128
    meta["wav_chan"] = meta["wav_band"] * meta["wav_band_chan"]
    meta["eeg_chan"] = meta["eeg_band"] * meta["eeg_band_chan"]
    meta["chan_num"] = meta["eeg_chan"] + meta["wav_chan"]

    return eeg, voice, label, meta


def preprocess_eeg(eeg, l_freq: int = 1, h_freq=50, ica=True, *args, **kwargs):
    # 脑电数据预处理（ICA）
    eeg = ica_eeg(eeg) if ica else eeg

    # 滤波过程， 采样率降低为128Hz
    eeg = filter_eeg(eeg, l_freq, h_freq)

    return eeg


def preprocess_voice(voice, label, l_freq: int = 1, h_freq=50, label_type='speaker', *args, **kwargs):
    # 语音数据预处理（希尔伯特变换、p-law、滤波）
    voice = filter_voice(voice, l_freq=l_freq, h_freq=h_freq, *args, **kwargs)

    if label_type == "direction":
        pass
    elif label_type == "speaker":
        voice = voice_dirct2attd(voice, label)
        voice = voice_attd2speaker(voice, label)
    else:
        raise ValueError('“label_type”不属于已知（direction、speaker）')

    return voice


def data_loader(sub_id):
    """
    读取原始数据。
    输出的音频的第一个是左边，第二个是右边
    Args:
        sub_id: 需要提取的受试者编号

    """
    eeg, voice, label = [], [], []

    # 建立数据存储空间
    if dataset_name == 'SCUT':
        pass
    else:
        raise ValueError('数据库名称错误：“dataset_name”不属于已知数据库（SCUT）')

    fs_eeg = fs
    fs_voice = dtu_audio_fs

    # 数据裁剪
    for k_tra in range(len(eeg)):
        data_len = []
        data_len.append(int(eeg[k_tra].shape[0] / fs_eeg))
        data_len.append(int(len(voice[k_tra][0]) / fs_voice))
        data_len.append(int(len(voice[k_tra][1]) / fs_voice))
        data_len.append(55)

        # 计算最短的数据时长（秒）
        min_len = min(data_len)

        # 计算个样本的帧数
        eeg_len = min_len * fs_eeg
        voice_len = min_len * fs_voice

        eeg[k_tra] = eeg[k_tra][0:eeg_len, :]
        voice[k_tra][0] = voice[k_tra][0][0:voice_len]
        voice[k_tra][1] = voice[k_tra][1][0:voice_len]

    return eeg, voice, label


def filter_voice(voice, l_freq, h_freq, fs=128, is_hilbert=True, is_p_law=True, internal_fs=8000, gl=150, gh=4000,
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
    fs_voice = 44100

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


def ica_eeg(eeg):
    """
    对数据进行ICA处理，去伪迹，同时去除50Hz工频干扰
    Args:
        eeg: 原始输入数据
    Returns:
        data: 处理后的数据

    """

    ica_dict = [0, 1]  # 手动选择的ICA成分

    # 准备电极信息
    info = set_info()

    # 加载模板数据（S1-Trail1）
    eeg_tmp, voice_tmp, label_tmp = data_loader('1')
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


def filter_eeg(eeg, l_freq, h_freq):
    """
    对数据进行滤波处理，并降低采样率到128Hz（标准化的采样率）
    Args:
        eeg: 去伪迹后的数据
        l_freq:带通滤波的低频范围
        h_freq:带通滤波的高频范围

    Returns:
        data: 滤波后的数据

    """

    # 滤波
    is_verbose = True
    info = set_info()
    for k_tra in range(len(eeg)):
        print(f'data filter, trail: {k_tra}')

        my_eeg = eeg[k_tra]
        my_eeg = np.transpose(my_eeg, [1, 0])

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(my_eeg, info, verbose=is_verbose)

        # 重参考、滤波、降采样
        # TODO: 添加多频带处理机制
        raw = raw.set_eeg_reference(ref_channels='average', verbose=is_verbose)
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=is_verbose)
        raw = raw.resample(128)


        # 储存数据
        my_eeg = raw.get_data()[0:64, :]
        my_eeg = (my_eeg - np.average(my_eeg, axis=-1)[..., None]) / np.std(my_eeg, axis=-1)[..., None]
        my_eeg = np.transpose(my_eeg, [1, 0])
        eeg[k_tra] = my_eeg

        # 关闭可视化过程
        is_verbose = False

    return eeg


def set_info():
    """
    设置电极信号（用于mne的数据格式转换）

    Returns:
          info：通道数据等

    """

    ch_names = channel_names_scut
    ch_types = list(['eeg' for _ in range(len(ch_names))])

    info = mne.create_info(ch_names, fs, ch_types)
    info.set_montage(montage)

    return info


if __name__ == '__main__':
    data_eeg, data_voice, data_label, split_index = preprocess('SCUT', sub_id='16', l_freq=1, h_freq=50, is_ica=True, need_voice=False)