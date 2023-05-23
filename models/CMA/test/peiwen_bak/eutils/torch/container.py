#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   container.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/12 1:24   lintean      1.0         None
'''
from dataclasses import dataclass
from typing import Any, List
import numpy as np
from enum import auto, Enum
import torch
from torch.utils.data import Dataset
import math
import torch.nn.functional as func


class AADDataset(Enum):
    DTU: auto()
    KUL: auto()
    SCUT: auto()


class ConType(Enum):
    No: auto()
    Low: auto()
    High: auto()


@dataclass
class DataMeta:
    dataset: AADDataset
    subj_num: int
    trail_num: int
    trail_len: int
    con_type: List[ConType]
    eeg_fs: int = 128
    wav_fs: int = 128

    eeg_band: int = 1
    eeg_band_chan: int = 64
    eeg_chan: int = 64
    wav_band: int = 1
    wav_band_chan: int = 1
    wav_chan: int = 1
    chan_num: int = 66


@dataclass
class DecisionWindow:
    start: int
    end: int
    target: int
    win_idx: int
    trail_idx: int
    subj_idx: int


@dataclass
class AADData:
    meta: DataMeta
    wav1: List[np.ndarray] or np.ndarray
    wav2: List[np.ndarray] or np.ndarray
    eeg: List[np.ndarray] or np.ndarray
    labels: List[int] or np.ndarray
    tng_win: List[DecisionWindow]
    tes_win: List[DecisionWindow]


@dataclass
class SplitMeta:
    win_len: int = 128
    win_lap: int = None
    overlap: float = 0.0
    delay: int = 0
    cv_flod: int = 5
    cur_flod: int = 0
    tes_pct: float = 0.2
    valid_pct: float = 0.0


@dataclass
class TngMeta:
    batch_size: int = 32
    max_epoch: int = 100
    lr: float = 1e-3
    device: str = "cpu"


@dataclass
class AADModel:
    model: torch.nn.Module
    loss: torch.nn.Module
    optim: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.ChainedScheduler or None
    dev: torch.device

    def __init__(self, model, loss, optim, sched, dev):
        super(AADModel, self).__init__()
        self.model = model.to(dev)
        self.loss = loss.to(dev)
        self.optim = optim
        self.sched = sched
        self.dev = dev


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


class CurrSet(Dataset):
    def __init__(self, eeg: torch.Tensor, wav1: torch.Tensor, wav2: torch.Tensor, window: torch.Tensor, window_meta, args):
        super(CurrSet, self).__init__()
        self.eeg = eeg
        self.wav1 = wav1
        self.wav2 = wav2
        self.window = window
        self.wm = window_meta
        self.el = eeg.shape[-1]
        self.wl = wav1.shape[-1]
        self.args = args

    def __getitem__(self, index):
        start = self.window[index, self.wm.start]
        end = self.window[index, self.wm.end]
        fs_start = math.floor(start / self.el * self.wl) if self.el != self.wl else start
        wav_len = math.floor((end - start) / self.el * self.wl)
        fs_end = fs_start + wav_len
        if fs_end >= self.wav1.shape[-1]:
            fs_end = self.wav1.shape[-1] - 1
            fs_start = fs_end - wav_len
        wav1 = self.wav1[:, fs_start:fs_end]
        wav2 = self.wav2[:, fs_start:fs_end]
        eeg = self.eeg[:, start:end]
        label = self.window[index, self.wm.target] - 1

        if self.window.shape[-1] > 6 and "reversed" in self.wm:
            if self.window[index, self.wm.reversed] == 1:
                wav1, wav2 = wav2, wav1
                label = 1 - label

        return wav1, wav2, eeg, label, index

    def __len__(self):
        return self.window.shape[0]

    def call_epoch(self):
        if "random_reverse" in self.args and self.args.random_reverse:

            from eutils.split_utils import get_reverse
            reverse = get_reverse(self.window.shape[0])[..., None]
            reverse = torch.from_numpy(reverse).to(self.window.device)
            self.window = self.window[..., :-1]
            self.window = torch.concat([self.window, reverse], dim=-1)



class DataLabelSet(Dataset):
    def __init__(self, eeg: torch.Tensor, wav1: torch.Tensor, wav2: torch.Tensor, label: torch.Tensor):
        super(DataLabelSet, self).__init__()
        self.eeg = eeg
        self.wav1 = wav1
        self.wav2 = wav2
        self.label = label

    def __getitem__(self, index):
        return self.wav1[index], self.wav2[index], self.eeg[index], self.label[index], index

    def __len__(self):
        return self.eeg.shape[0]


