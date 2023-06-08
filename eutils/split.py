#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   split.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/16 22:15   lintean      1.0         None
'''
import copy
import math
import os
import random
import sys
from typing import List, Dict
import numpy as np
import pandas as pd
from dotmap import DotMap
from eutils.container import AADData, SplitMeta, DecisionWindow, DataMeta
from eutils.util import read_json


def preproc(args: DotMap, local: DotMap, **kwargs) -> tuple[AADData, DotMap, DotMap]:
    """
    从恩泽师兄开发的预处理程序中读取EEG、语音以及相关的meta
    读取EEG+语音
    @param data: 占位符
    @param args: 全局meta
    @param local: subject个人meta
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [ConType * trial, time, channel]
    """
    from eutils.preproc.db_preprocess import preprocess as preprocess

    datasets = ['DTU', 'KUL', 'SCUT']
    label_types = ["direction", "speaker"]
    for dataset in datasets:
        if dataset in args.data_name:
            pm = local.preproc_meta
            eeg, audio, labels, meta = preprocess(dataset, local.name[1:], **pm.__dict__)
            meta = DataMeta(**meta)
            local.data_meta = meta
            data = AADData(eeg=eeg, audio=audio, labels=labels, meta=meta)
            return data, args, local


def read_data(args: DotMap, local: DotMap, **kwargs) -> tuple[AADData, DotMap, DotMap]:
    """
    从已经预处理好的数据中读取EEG、语音以及相关的meta
    读取EEG+语音
    @param data: 占位符
    @param args: 全局meta
    @param local: subject个人meta
    @return:
    """

    path = args.data_document_path

    # read metadata
    dm = read_json(f"{path}/metadata.json")
    meta = DataMeta(
        dataset=args.database.name,
        subj_id=local.name[1:],
        trail_num=dm['trail_number'],
        con_type=args.ConType,
        eeg_fs=dm['fs'] if 'eeg_fs' not in dm else dm['eeg_fs'],
        wav_fs=dm['fs'] if 'audio_fs' not in dm else dm['audio_fs'],
        eeg_band=dm['eeg_band'],
        eeg_band_chan=dm['eeg_channel_per_band'],
        wav_band=dm['audio_band'],
        wav_band_chan=dm['audio_channel_per_band'],
    )

    # read data
    # 将不同声学环境的数据当成额外增加的trail
    eeg, audio, labels = [], [], []
    for con_type in meta.con_type:
        label = pd.read_csv(f"{path}/csv/{local.name}{con_type.value}.csv")
        label = label.to_numpy()
        for k in range(meta.trail_num):
            wav_file = f"{path}/{con_type.value}/{local.name}Tra{k + 1}_audio.csv"
            filename = f"{path}/{con_type.value}/{local.name}Tra{k + 1}.csv"
            data_pf = pd.read_csv(filename, header=None)

            if os.path.exists(wav_file):
                wav = pd.read_csv(wav_file, header=None)
                EEG_data = data_pf
            else:
                wav = data_pf
                EEG_data = data_pf.iloc[:, 2 * meta.wav_chan:]

            wav1 = wav.iloc[:, 0:meta.wav_chan]
            wav2 = wav.iloc[:, meta.wav_chan:meta.wav_chan * 2]

            eeg.append(EEG_data.to_numpy(dtype=np.float32))
            # DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
            if "KUL" == args.database.name:
                labels.append({
                    'direction': label[k][0] - 1,
                    'speaker': label[k][1] - 1
                })
            elif "DTU" == args.database.name:
                labels.append({
                    'direction': label[k][1] - 1,
                    'speaker': label[k][0] - 1
                })
            else:
                raise ValueError('数据库不属于已知（KUL、DTU）')

            if local.preproc_meta.need_voice:
                # 如果是KUL 先把attend换回来
                if "KUL" == args.database.name:
                    wav1, wav2 = wav2, wav1

                audio.append([
                    wav1.to_numpy(dtype=np.float32),
                    wav2.to_numpy(dtype=np.float32)
                ])

    if not local.preproc_meta.need_voice:
        audio = None

    local.data_meta = meta
    data = AADData(eeg=eeg, audio=audio, pre_lbl=labels, meta=meta)
    return data, args, local


def ds_labels(audio: List[List[np.ndarray]], pre_lbl: List[Dict], label_type: str):
    labels = []
    for i in range(len(pre_lbl)):
        labels.append(pre_lbl[i][label_type])
        if audio is not None and pre_lbl[i][label_type] == 1:
            audio[i][0], audio[i][1] = audio[i][1], audio[i][0]

    return audio, labels


def select_labels(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    audio, labels = ds_labels(data.audio, data.pre_lbl, local.preproc_meta.label_type)
    data.audio = audio
    data.labels = labels

    return data, args, local


def window_split(data: AADData, split_meta: SplitMeta, **kwargs):
    """
    对信号进行分窗
    Args:
        x: list[np.ndarray], len(x) is trails_num, each element shape as [time, channel]

    Returns: 分窗后的数据 shape as [sample, time, channel]

    """
    win_lap = math.floor(split_meta.time_lap * data.meta.eeg_fs)
    win_len = math.floor(split_meta.time_len * data.meta.eeg_fs)
    windows = []
    # 对于每个trail进行划分
    for trail_idx in range(len(data.eeg)):
        trail_len = data.eeg[trail_idx].shape[0]
        left = np.arange(start=0, stop=trail_len, step=win_lap)
        while left[-1] + win_len - 1 > trail_len - 1:
            left = np.delete(left, -1, axis=0)
        right = left + win_len

        trail_win = []

        for idx in range(len(left)):
            trail_win.append(DecisionWindow(
                trail_idx=trail_idx,
                start=left[idx],
                end=right[idx],
                label=data.labels[trail_idx],
                win_idx=len(trail_win),
                subj_idx=data.meta.subj_id
            ))
        windows.append(trail_win)

    return windows


def trails_split(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    data.windows = window_split(data=data, split_meta=local.split_meta)
    count = 0
    for win in data.windows:
        count += len(win)
    local.logger.info(f'the count of window: {count}')
    return data, args, local


def cv_divide(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    if "current_flod" not in local:
        if "current_flod" in args and args.cross_validation_fold > 1:
            local.current_flod = args.current_flod
        else:
            local.current_flod = 0
    tng_win, tes_win = cv_func(data=data, split_meta=args.split_meta, random_seed=args.random_seed)
    data.tng_win = tng_win
    data.tes_win = tes_win
    local.logger.info(f'{args.split_meta.curr_flod} of cv{args.split_meta.cv_flod}')
    return data, args, local


def cv_func(data: AADData, split_meta: SplitMeta, **kwargs) -> tuple[List[DecisionWindow], List[DecisionWindow]]:
    tng_win = []
    tes_win = []
    # 对于每个trail进行划分
    for trail_idx in range(len(data.eeg)):
        trail_len = data.eeg[trail_idx].shape[0]

        # 随机抽取的测试窗口长度和左右边界
        tes_win_len = trail_len / split_meta.cv_flod
        tes_win_left = math.floor(trail_len * split_meta.curr_flod / split_meta.cv_flod)
        tes_win_right = tes_win_left + tes_win_len - 1

        trail_win = data.windows[trail_idx]
        for idx in range(len(trail_win)):
            if trail_win[idx].end < tes_win_left or trail_win[idx].start > tes_win_right:
                tng_win.append(trail_win[idx])
            elif tes_win_left <= trail_win[idx].start < trail_win[idx].end <= tes_win_right:
                tes_win.append(trail_win[idx])

    return tng_win, tes_win


def hold_on_divide(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    tng_win, tes_win = hold_on_func(data=data, split_meta=local.split_meta, random_seed=args.random_seed)
    data.tng_win = tng_win
    data.tes_win = tes_win
    local.logger.info(f'the count of tng_win: {len(tng_win)}')
    local.logger.info(f'the count of tes_win: {len(tes_win)}')
    return data, args, local


def hold_on_func(data: AADData, split_meta: SplitMeta, random_seed, **kwargs) -> tuple[List[DecisionWindow], List[DecisionWindow]]:
    random.seed(random_seed)
    tng_win = []
    tes_win = []
    # 对于每个trail进行划分
    for trail_idx in range(len(data.eeg)):
        trail_len = data.eeg[trail_idx].shape[0]

        # 随机抽取的测试窗口长度和左右边界
        tes_win_len = math.floor(trail_len * split_meta.tes_pct)
        tes_win_left = random.randint(0, trail_len - tes_win_len)
        tes_win_right = tes_win_left + tes_win_len - 1

        trail_win = data.windows[trail_idx]
        for idx in range(len(trail_win)):
            if trail_win[idx].end < tes_win_left or trail_win[idx].start > tes_win_right:
                tng_win.append(trail_win[idx])
            elif tes_win_left <= trail_win[idx].start < trail_win[idx].end <= tes_win_right:
                tes_win.append(trail_win[idx])

    return tng_win, tes_win


def neg_samples_add(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    def __neg_samples_add(windows: List[DecisionWindow]) -> List[DecisionWindow]:
        neg_win = []
        for idx in range(len(windows)):
            temp = copy.deepcopy(windows[idx])
            temp.reversed = 1
            temp.win_idx += len(windows)
            neg_win.append(temp)
            windows[idx].reversed = 0
        return windows + neg_win

    data.tng_win = __neg_samples_add(data.tng_win)
    data.tes_win = __neg_samples_add(data.tes_win)
    local.logger.info(f"add negative samples to training window, current count: {len(data.tng_win)}")
    local.logger.info(f"add negative samples to test window, current count: {len(data.tes_win)}")
    return data, args, local


def rept_win_remove(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    delete_axis = []
    for j in range(len(data.tng_win)):
        for i in range(len(data.tes_win)):
            if data.tng_win[j].start < data.tes_win[i].start < data.tng_win[j].end or \
                    data.tng_win[j].start < data.tes_win[i].end < data.tng_win[j].end:
                delete_axis.append(j)
    local.logger.info(f"remove repeated training window: {delete_axis}")

    for idx in delete_axis[::-1]:
        data.tng_win.pop(idx)
    return data, args, local


