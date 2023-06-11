#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   KUL.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/17 15:40   lintean      1.0         None
'''
name = "KUL"

subj_number = 16
trail_number = 8

# eeg采样率
kul_eeg_fs = 128

# 语音采样率
kul_audio_fs = 44100

# 方向和讲话者
# 预处理时从文件中读取
kul_label = []

# kul通道顺序
# 预处理时从mne中获取
channel_names_kul = []

# 数据库状态
KULMeta = {
    'dataset': 'KUL',
    'subj_id': "1",
    'trail_num': trail_number,
    'con_type': ['No'],
    'eeg_fs': kul_eeg_fs,
    'wav_fs': kul_audio_fs,
    'eeg_band': 1,
    'eeg_band_chan': 64,
    'eeg_chan': 64,
    'wav_band': 1,
    'wav_band_chan': 1,
    'wav_chan': 1,
    'chan_num': 66
}