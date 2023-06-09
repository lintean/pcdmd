#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/24 10:00   lintean      1.0         None
'''

import copy
import os
from typing import List

import mne
import numpy as np
import scipy.io
from scipy.io import wavfile
from .KUL import kul_eeg_fs, kul_audio_fs, kul_label, trail_number, channel_names_scut, KULMeta
from ..prep_util import ica_eeg, filter_eeg, filter_voice, voice_dirct2attd, voice_attd2speaker

ica_dict = [0, 2]
montage = 'biosemi64'
fs = kul_eeg_fs


def preprocess(dataset_path="../AAD_KUL", sub_id="1", eeg_lf: int or List[int] = 1, eeg_hf: int or List[int] = 32,
                wav_lf: int or List[int] = 1, wav_hf: int or List[int] = 32,
               ica=True, label_type='speaker', need_voice=True, *args, **kwargs):
    """
    读取数据库的数据，经过ICA去伪迹，然后带通滤波，最后输出标准的样本及标签
    Args:
        dataset_path: 数据库路径
        sub_id: 需要提取的受试者编号
        eeg_lf: 带通滤波器参数，低频下限
        eeg_hf: 带通滤波器参数，高通上限
        wav_lf：
        wav_hf：
        ica: 是否进行ICA处理
        label_type：
        need_voice：

    Returns:
        eeg：list，each item is a trail: np.array[Time*Channels]
        voice：list，each item is a trail: list[np.array(Time*Channels)], the first one is attend
        label：list，each item is a trail: list[方位、讲话者]，以0、1标记

    """

    if isinstance(eeg_lf, int) and isinstance(eeg_hf, int):
        eeg_lf, eeg_hf = [eeg_lf], [eeg_hf]
    if isinstance(wav_lf, int) and isinstance(wav_hf, int):
        wav_lf, wav_hf = [wav_lf], [wav_hf]

    # 加载数据
    eeg, voice, label = data_loader(dataset_path, sub_id)
    meta = KULMeta

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
        eegs.append(preprocess_eeg(dataset_path, eeg.copy(), eeg_lf[idx], eeg_hf[idx], ica, *args, **kwargs))
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

    meta["subj_id"] = sub_id
    meta["eeg_fs"] = 128
    meta["wav_fs"] = 128
    meta["wav_chan"] = meta["wav_band"] * meta["wav_band_chan"]
    meta["eeg_chan"] = meta["eeg_band"] * meta["eeg_band_chan"]
    meta["chan_num"] = meta["eeg_chan"] + meta["wav_chan"]

    return eeg, voice, label, meta


def preprocess_eeg(dataset_path, eeg, l_freq: int = 1, h_freq=50, ica=True, *args, **kwargs):
    # 脑电数据预处理（ICA）
    if ica:
        eeg_tmp, _, _ = data_loader(dataset_path, "1")
        eeg = ica_eeg(eeg, ica_dict=ica_dict, info=set_info(), montage=montage, eeg_tmp=eeg_tmp)

    # 滤波过程， 采样率降低为128Hz
    eeg = filter_eeg(eeg, info=set_info(), ref_channels='average', l_freq=l_freq, h_freq=h_freq)

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


def data_loader(dataset_path, sub_id):
    """
    读取原始数据。
    输出的音频的第一个是左边，第二个是右边
    Args:
        dataset_path: 数据库路径
        sub_id: 需要提取的受试者编号

    """
    eeg, voice, label = [], [], []

    data_path = f'{dataset_path}/S{sub_id}.mat'
    data_mat = scipy.io.loadmat(data_path)

    for k_tra in range(trail_number):
        # 加载语音数据[左侧音频，右侧音频]
        tmp_voice = []
        for k_voice in range(2):
            voice_file = data_mat['trials'][0, k_tra]['stimuli'][0][0][k_voice][0][0]
            voice_path = f'{dataset_path}/stimuli/{voice_file}'
            tmp_voice.append(wavfile.read(voice_path)[1])  # 加载语音数据
        # 合并脑电数据
        voice.append(tmp_voice)

        # 加载脑电数据
        tmp_eeg = data_mat['trials'][0, k_tra]['RawData'][0, 0]['EegData'][0, 0]
        eeg.append(tmp_eeg)

        # 加载标签数据[左右，讲话者编号]
        lab = 0 if str(data_mat['trials'][0, k_tra]['attended_ear'][0, 0][0]) == 'L' else 1
        tmp_label = {
            'direction': lab,
            'speaker': data_mat['trials'][0, k_tra]['attended_track'][0][0][0][0] - 1
        }
        label.append(tmp_label)

    fs_eeg = fs
    fs_voice = kul_audio_fs

    # 数据裁剪
    for k_tra in range(len(eeg)):
        data_len = []
        data_len.append(int(eeg[k_tra].shape[0] / fs_eeg))
        data_len.append(int(len(voice[k_tra][0]) / fs_voice))
        data_len.append(int(len(voice[k_tra][1]) / fs_voice))
        data_len.append(360)

        # 计算最短的数据时长（秒）
        min_len = min(data_len)

        # 计算个样本的帧数
        eeg_len = min_len * fs_eeg
        voice_len = min_len * fs_voice

        eeg[k_tra] = eeg[k_tra][0:eeg_len, :]
        voice[k_tra][0] = voice[k_tra][0][0:voice_len]
        voice[k_tra][1] = voice[k_tra][1][0:voice_len]

    return eeg, voice, label


def set_info():
    """
    设置电极信号（用于mne的数据格式转换）

    Returns:
          info：通道数据等

    """

    ch_names = mne.channels.make_standard_montage(montage).ch_names
    ch_types = list(['eeg' for _ in range(len(ch_names))])

    info = mne.create_info(ch_names, fs, ch_types)
    info.set_montage(montage)

    return info


if __name__ == '__main__':
    pass