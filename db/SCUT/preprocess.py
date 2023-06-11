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
from scipy.io import wavfile
from .SCUT import scut_eeg_fs, scut_audio_fs, scut_label, trail_number, channel_names_scut, SCUTMeta, scut_remove_trail
from .util import *
from ..prep_util import ica_eeg, filter_eeg, filter_voice, voice_dirct2attd, voice_attd2speaker

ica_dict = [0, 1]
montage = 'standard_1020'
fs = scut_eeg_fs


def preprocess(
        sub_id="1",
        data_path="../AAD_SCUT",
        eeg_lf: int or List[int] = 1, 
        eeg_hf: int or List[int] = 32,
        wav_lf: int or List[int] = 1, 
        wav_hf: int or List[int] = 32,
        ica=True, 
        label_type='speaker', 
        need_voice=True, 
        *args, 
        **kwargs
):
    """
    读取数据库的数据，经过ICA去伪迹，然后带通滤波，最后输出标准的样本及标签
    Args:
        sub_id: 需要提取的受试者编号
        data_path: 数据库路径
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
    eeg, voice, label = data_loader(data_path, sub_id)
    meta = SCUTMeta

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
        eegs.append(preprocess_eeg(data_path, eeg.copy(), eeg_lf[idx], eeg_hf[idx], ica, *args, **kwargs))
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
    if sub_id in scut_remove_trail:
        meta["trail_num"] = meta["trail_num"] - len(scut_remove_trail[sub_id])
    meta["eeg_fs"] = 128
    meta["wav_fs"] = 128
    meta["wav_chan"] = meta["wav_band"] * meta["wav_band_chan"]
    meta["eeg_chan"] = meta["eeg_band"] * meta["eeg_band_chan"]
    meta["chan_num"] = meta["eeg_chan"] + meta["wav_chan"]

    return eeg, voice, label, meta


def preprocess_eeg(data_path, eeg, l_freq: int = 1, h_freq: int = 32, ica: bool = True, *args, **kwargs):
    # 脑电数据预处理（ICA）
    if ica:
        eeg_tmp, _, _ = data_loader(data_path, "1")
        eeg = ica_eeg(eeg, ica_dict=ica_dict, info=set_info(), montage=montage, eeg_tmp=eeg_tmp)

    # 滤波过程， 采样率降低为128Hz
    eeg = filter_eeg(eeg, info=set_info(), ref_channels='average', l_freq=l_freq, h_freq=h_freq)

    return eeg


def preprocess_voice(voice, label, l_freq: int = 1, h_freq: int = 32, label_type: str = 'speaker', *args, **kwargs):
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


def data_loader(data_path, sub_id):
    """
    读取原始数据。
    输出的音频的第一个是左边，第二个是右边
    Args:
        data_path: 数据库路径
        sub_id: 需要提取的受试者编号

    """
    eeg, voice, label = [], [], []

    # 加载语音数据
    for k_tra in range(trail_number):
        tmp_voice = []
        voice_path = f'{data_path}/clean/Trail{int(k_tra + 1)}.wav'
        tmp, my_voice = wavfile.read(voice_path)
        my_voice = np.array(my_voice)
        for k_voice_track in range(2):
            voice_track = my_voice[:, k_voice_track]
            voice_track = voice_track.tolist()
            while voice_track[-1] == 0:
                voice_track.pop()
            tmp_voice.append(voice_track)
        voice.append(tmp_voice)

    # 加载脑电数据
    file_path = f'{data_path}/S{sub_id}/'
    files = os.listdir(file_path)
    files = sorted(files)  # 按顺序，避免label不同
    for file in files:
        # 输入格式化
        data_mat = scipy.io.loadmat(file_path + file)
        for k_tra in range(data_mat['Markers'].shape[1] // 3):
            k_sta = data_mat['Markers'][0, 3 * k_tra + 2][3][0][0]
            # 避免Trail中断
            if len(data_mat['Markers'][0]) > 3 * k_tra + 3:
                k_end = data_mat['Markers'][0, 3 * k_tra + 3][3][0][0]
            else:
                k_end = len(data_mat[channel_names_scut[0]]) - 1

            tmp_eeg = np.zeros((k_end - k_sta, 64))
            for k_cha in range(len(channel_names_scut)):
                tmp_eeg[:, k_cha] = data_mat[channel_names_scut[k_cha]][k_sta:k_end, 0]
            eeg.append(tmp_eeg)

    label = copy.deepcopy(scut_label)
    voice, label = scut_order(voice, label, sub_id)

    # 处理异常的Trail，比如不完全的Trail等
    eeg, voice, label = scut_remove(eeg, voice, label, sub_id)


    fs_eeg = fs
    fs_voice = scut_audio_fs

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
    pass