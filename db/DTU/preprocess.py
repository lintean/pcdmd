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
from .DTU import dtu_eeg_fs, dtu_audio_fs, DTUMeta
from ..prep_util import ica_eeg, filter_eeg, filter_voice, voice_dirct2attd, voice_attd2speaker

ica_dict = [0, 1, 7]
montage = 'biosemi64'
fs = dtu_eeg_fs


def preprocess(
        sub_id="1",
        data_path="../AAD_KUL",
        con_type=None,
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
        con_type: 需要读取的声学环境
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
        meta: meta
    """

    if con_type is None:
        con_type = ["No"]
    if isinstance(eeg_lf, int) and isinstance(eeg_hf, int):
        eeg_lf, eeg_hf = [eeg_lf], [eeg_hf]
    if isinstance(wav_lf, int) and isinstance(wav_hf, int):
        wav_lf, wav_hf = [wav_lf], [wav_hf]

    # 加载数据
    eeg, voice, label = data_loader(data_path, sub_id, con_type)
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
        eegs.append(preprocess_eeg(data_path, eeg.copy(), con_type, eeg_lf[idx], eeg_hf[idx], ica, *args, **kwargs))
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


def preprocess_eeg(data_path, eeg, con_type, l_freq: int = 1, h_freq: int = 32, ica: bool = True, *args, **kwargs):
    # 脑电数据预处理（ICA）
    if ica:
        eeg_tmp, _, _ = data_loader(data_path, "1", con_type)
        eeg = ica_eeg(eeg, ica_dict=ica_dict, info=set_info(), montage=montage, eeg_tmp=eeg_tmp)

    # 滤波过程， 采样率降低为128Hz
    eeg = filter_eeg(eeg, info=set_info(), ref_channels=['ecg1', 'ecg2'], l_freq=l_freq, h_freq=h_freq)

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


def data_loader(data_path, sub_id, con_type):
    """
    读取原始数据。
    输出的音频的第一个是左边，第二个是右边
    Args:
        data_path: 数据库路径
        sub_id: 需要提取的受试者编号
        con_type: 需要读取的声学环境
    """
    eeg, voice, label = [], [], []

    # 加载数据
    file = f'{data_path}/EEG_new/S{sub_id}.mat'
    data_mat = scipy.io.loadmat(file)

    # 划分不同的Trials
    data_all_trails = data_mat['data'][0, 0]['eeg'][0, 0]
    event_index = data_mat['data'][0, 0]['event'][0, 0]['eeg'][0, 0]['sample']
    n_speakers = data_mat['data'][0, 0]['n_speakers']
    male_list = data_mat['data'][0, 0]['wavfile_male']
    female_list = data_mat['data'][0, 0]['wavfile_female']
    acoustic_condition = data_mat['data'][0, 0]['acoustic_condition']

    # 去除EOG通道
    data_all_trails = data_all_trails[:, :66]

    # 根据声学环境提取数据
    for con in con_type:

        # 计算正式实验的索引
        n_speakers = np.array(n_speakers)
        idxs = np.where(
            [acoustic_condition[i, 0] == con.value and n_speakers[i, 0] == 2 for i in range(n_speakers.shape[0])]
        )

        # 加载数据
        for k_tra in idxs[0]:
            # 读取标签信息[左右，讲话者编号]
            tmp_label = {
                'direction': data_mat['data'][0, 0]['attend_lr'][k_tra, 0] - 1,
                'speaker': data_mat['data'][0, 0]['attend_mf'][k_tra, 0] - 1
            }
            label.append(tmp_label)

            # 加载语音数据
            male_name = male_list[k_tra][0][0]
            female_name = female_list[k_tra][0][0]
            tmp_voice = []
            voice_file_list = [female_name, male_name] if tmp_label['direction'] + tmp_label['speaker'] == 1 else [male_name, female_name]
            for voice_file_name in voice_file_list:
                voice_path = f'{data_path}/AUDIO/{voice_file_name}'
                tmp_voice.append(wavfile.read(voice_path)[1])
            voice.append(tmp_voice)

            # 读取脑电数据
            ind_s, ind_e = event_index[2 * k_tra, 0], event_index[2 * k_tra + 1, 0]
            tmp_eeg = data_all_trails[ind_s:ind_e, 0:66]
            eeg.append(tmp_eeg)

    fs_eeg = fs
    fs_voice = dtu_audio_fs

    # 数据裁剪
    for k_tra in range(len(eeg)):
        data_len = []
        data_len.append(int(eeg[k_tra].shape[0] / fs_eeg))
        data_len.append(int(len(voice[k_tra][0]) / fs_voice))
        data_len.append(int(len(voice[k_tra][1]) / fs_voice))
        data_len.append(49)

        # 计算最短的数据时长（秒）
        min_len = min(data_len)

        # 计算个样本的帧数
        eeg_len = min_len * fs_eeg
        voice_len = min_len * fs_voice

        eeg[k_tra] = eeg[k_tra][0:eeg_len, :]
        voice[k_tra][0] = voice[k_tra][0][0:voice_len]
        voice[k_tra][1] = voice[k_tra][1][0:voice_len]

    return eeg, voice, label


def set_info(is_add=False):
    """
    设置电极信号（用于mne的数据格式转换）

    Returns:
          info：通道数据等

    """

    ch_names = mne.channels.make_standard_montage(montage).ch_names
    ch_types = list(['eeg' for _ in range(len(ch_names))])


    ch_names = ch_names + ['ecg1', 'ecg2']
    ch_types = ch_types + ['ecg', 'ecg']
    if is_add:
        ch_names = ch_names + ['ecg1', 'ecg2', 'eog1', 'eog2', 'eog3', 'eog4', 'eog5', 'eog6']
        ch_types = ch_types + ['ecg', 'ecg', 'eog', 'eog', 'eog', 'eog', 'eog', 'eog']

    info = mne.create_info(ch_names, fs, ch_types)
    info.set_montage(montage)

    return info


if __name__ == '__main__':
    pass