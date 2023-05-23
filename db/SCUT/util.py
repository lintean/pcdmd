#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   util.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/28 20:37   lintean      1.0         None
'''
import numpy as np


def voice_dirct2attd(voice, label):
    for i in range(len(voice)):
        if label[i][0] == 1:
            voice[i][0], voice[i][1] = voice[i][1], voice[i][0]
    return voice


def voice_attd2speaker(voice, label):
    for i in range(len(voice)):
        if label[i][1] == 1:
            voice[i][0], voice[i][1] = voice[i][1], voice[i][0]
    return voice


def select_label(label, label_type):
    nlabel = []
    label_index = None
    if label_type == "direction":
        label_index = 0
    elif label_type == "speaker":
        label_index = 1
    else:
        raise ValueError('“label_type”不属于已知（direction、speaker）')

    for i in range(len(label)):
        nlabel.append(label[i][label_index])
    return nlabel


def _adjust_order(x, sub_id):
    from .SCUT import scut_suf_order
    if int(sub_id) > 10:
        nx = []
        for k_tra in scut_suf_order:
            nx.append(x[k_tra])
        x = nx
    return x


def _reverse_label(label, sub_id):
    if int(sub_id) > 10:
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = 1 - label[i][j]
    return label


def scut_order(voice, label, sub_id):
    """
    前10名和后10名的音频顺序不同，通过这个恢复后10名的对应的脑电信号
    受试者编号是从1开始的，所以S10也不需要重新编码
    后10名的关注也不同！
    Args:
        voice:
        label:
        sub_id:

    Returns:

    """
    label = _adjust_order(label, sub_id)
    label = _reverse_label(label, sub_id)
    if voice == None:
        return label
    else:
        voice = _adjust_order(voice, sub_id)
        return voice, label


def scut_remove(eeg, voice, label, sub_id):
    """
    处理异常的Trail，比如不完全的Trail等
    Args:
        eeg:
        voice:
        label:

    Returns:

    """
    from .SCUT import scut_remove_trail
    if isinstance(eeg, list):
        if sub_id in scut_remove_trail:
            remove_index = scut_remove_trail[sub_id]
            remove_index = sorted(remove_index, reverse=True)

            print('Attention: 删除特定索引')
            print(remove_index)

            # 删除异常的Trail
            for k_pop in remove_index:
                eeg.pop(k_pop - 1)
                if voice is not None:
                    voice.pop(k_pop - 1)
                label.pop(k_pop - 1)
    elif isinstance(eeg, np.ndarray):
        if sub_id in scut_remove_trail:
            keep_index = list(range(32))
            remove_index = scut_remove_trail[sub_id]

            print('Attention: 删除指定索引')
            print(remove_index)

            # 删除异常的Trail
            for k_pop in reversed(remove_index):
                keep_index.pop(k_pop - 1)
                label.pop(k_pop - 1)
            eeg = eeg[keep_index]
    if voice is not None:
        return eeg, voice, label
    else:
        return eeg, label


def trails_split(x, time_len, window_lap):
    """
    对信号进行分窗
    Args:
        x: list[np.ndarray], len(x) is trails_num, each element shape as [time, channel]
        time_len:
        window_lap:

    Returns: 分窗后的数据 shape as [sample, time, channel]

    """
    sample_len = int(128 * time_len)

    split_index = []
    split_x = []
    # 对于每个trail进行划分
    for i_tra in range(len(x)):
        trail_len = x[i_tra].shape[0]
        left = np.arange(start=0, stop=trail_len, step=window_lap)
        while left[-1] + sample_len - 1 > trail_len - 1:
            left = np.delete(left, -1, axis=0)
        right = left + sample_len

        split_index.append(np.stack([left, right], axis=-1))

        temp = [x[i_tra][left[i]: right[i]] for i in range(left.shape[0])]
        split_x.append(np.stack(temp, axis=0))

    return split_x, split_index


def list_data_split(eeg, voice, label, time_len, band_num, window_lap):
    """
    对数据进行标准化，变成N*P*C，P为样本长度（帧数），C为通道数量
    Args:
        voice: 语音
        eeg:处理后的干净数据（Trails*Times*Channels）
        label:标签
        time_len:样本时长（秒）
        band_num: 频带数量

    Returns:
        eeg：格式化数据
        label：格式化标签

    """
    # eeg和voice分窗
    if voice is not None:
        for i in range(len(voice)):
            voice[i] = np.stack(voice[i], axis=-1)
            if voice[i].shape[0] != eeg[i].shape[0]:
                raise ValueError("the length of voice and eeg must be the same")
        voice, split_index = trails_split(voice, time_len, window_lap)
    eeg, split_index = trails_split(eeg, time_len, window_lap)

    # label使用one-hot encoding
    total_label = []
    for i_tra in range(len(eeg)):
        samples_num = eeg[i_tra].shape[0]
        sub_label = label[i_tra] * np.ones(samples_num)
        sub_label = np.eye(2)[sub_label.astype(int)]
        total_label.append(sub_label)

    label = total_label

    if voice is not None:
        return eeg, voice, label, split_index
    else:
        return eeg, label, split_index
