#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DTU.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/17 15:41   lintean      1.0         None
'''
name = "DTU"

subj_number = 18
trail_number = 20

# eeg采样率
dtu_eeg_fs = 128

# 语音采样率
dtu_audio_fs = 44100

# 方向和讲话者
# 预处理时从文件中读取
dtu_label = []

# kul通道顺序
# 预处理时从mne中获取
channel_names_scut = []

# 数据库状态
DTUMeta = {
    'dataset': 'DTU',
    'subj_id': "1",
    'trail_num': trail_number,
    'con_type': ['No', 'Low', 'High'],
    'eeg_fs': dtu_eeg_fs,
    'wav_fs': dtu_audio_fs,
    'eeg_band': 1,
    'eeg_band_chan': 64,
    'eeg_chan': 64,
    'wav_band': 1,
    'wav_band_chan': 1,
    'wav_chan': 1,
    'chan_num': 66
}