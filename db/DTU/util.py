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