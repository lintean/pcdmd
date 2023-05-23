#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   container.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/12 1:24   lintean      1.0         None
'''
from dataclasses import dataclass
from typing import Any, List, Dict
import numpy as np
from enum import auto, Enum
import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as func


class AADDataset(Enum):
    DTU: str = "DTU"
    KUL: str = "KUL"
    SCUT: str = "SCUT"


class ConType(Enum):
    No: str = "No"
    Low: str = "Low"
    High: str = "High"


@dataclass
class DataMeta:
    dataset: AADDataset = None
    subj_id: str = None
    trail_num: int = None
    con_type: List[ConType] = None
    eeg_fs: int = None
    wav_fs: int = None

    eeg_band: int = 1
    eeg_band_chan: int = 64
    eeg_chan: int = None
    wav_band: int = 1
    wav_band_chan: int = 1
    wav_chan: int = None
    chan_num: int = None

    def __post_init__(self):
        if not isinstance(self.dataset, AADDataset):
            self.dataset = AADDataset(self.dataset)

        for i in range(len(self.con_type)):
            if not isinstance(self.con_type[i], ConType):
                self.con_type[i] = ConType(self.con_type[i])

        self.eeg_chan = self.eeg_band * self.eeg_band_chan
        self.wav_chan = self.wav_band * self.wav_band_chan
        self.chan_num = self.eeg_chan + self.wav_chan * 2

    def __iter__(self):
        return self.__dict__


@dataclass
class DecisionWindow:
    trail_idx: int = None
    start: int = None
    end: int = None
    label: int = None
    win_idx: int = None
    subj_idx: str = None
    reversed: bool = None

    def __post_init__(self):
        self.trail_idx = int(self.trail_idx)
        self.start = int(self.start)
        self.end = int(self.end)
        self.label = int(self.label)
        self.win_idx = int(self.win_idx)


@dataclass
class SplitMeta:
    time_len: float = 1
    time_lap: float = None
    overlap: float = 0.0
    delay: int = 0
    cv_flod: int = None
    curr_flod: int = None
    tes_pct: float = 0.2
    valid_pct: float = 0.0

    def __post_init__(self):
        if self.overlap is not None:
            self.time_lap = self.time_len * (1 - self.overlap)

    def __iter__(self):
        return self.__dict__


@dataclass
class AADModel:
    model: torch.nn.Module
    loss: torch.nn.Module
    optim: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.ChainedScheduler or None
    warmup: Any
    dev: torch.device

    def __init__(self, model, loss, optim, sched=None, warmup=None, dev=torch.device("cpu")):
        super().__init__()
        self.model = model.to(dev)
        self.loss = loss.to(dev)
        self.optim = optim
        self.sched = sched
        self.warmup = warmup
        self.dev = dev


@dataclass
class AADData:
    meta: DataMeta = None
    # 默认第一个是attend
    audio: List[List[np.ndarray]] or np.ndarray = None
    eeg: List[np.ndarray] or np.ndarray = None
    # pre_lbl和labels 取值从0开始
    pre_lbl: List[Dict] = None
    labels: List[int] or np.ndarray = None
    windows: List[List[DecisionWindow]] = None
    tng_win: List[DecisionWindow] = None
    tes_win: List[DecisionWindow] = None
    tng_loader: DataLoader = None
    tes_loader: DataLoader = None
    aad_model: AADModel = None

    def __post_init__(self):
        if isinstance(self.meta, Dict):
            self.meta = DataMeta(**self.meta)

    def __iter__(self):
        return self.__dict__


@dataclass
class PreprocMeta:
    eeg_lf: int or List[int] = 1
    eeg_hf: int or List[int] = 32
    wav_lf: int or List[int] = 1
    wav_hf: int or List[int] = 32
    label_type: str = "direction"
    need_voice: bool = False
    ica: bool = True
    internal_fs: int = 8000
    gl: int = 150
    gh: int = 4000
    space: float = 1.5
    is_combine: bool = True

    def __iter__(self):
        return self.__dict__


@dataclass
class TngMeta:
    batch_size: int = 32
    max_epoch: int = 100
    lr: float = 1e-3
    device: str = "cpu"

    def __iter__(self):
        return self.__dict__


@dataclass
class DataSplit:
    meta: SplitMeta
    split_steps: list


@dataclass
class AADTrain:
    data: AADData
    split: DataSplit
    model: List[AADModel] or AADModel
    meta: TngMeta
    train_steps: list


@dataclass
class AAD:
    label: str
    random_seed: int
    dataset_path: str
    subj_id: str
    train: AADTrain


@dataclass
class MultAAD:
    subjects: List[AAD]



