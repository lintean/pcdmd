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
import keras
import math


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
    model: Any
    loss: Any
    optim: Any
    sched: Any


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


class DataGenerator(keras.utils.Sequence):
    def __init__(self, eeg: np.ndarray, wav1: np.ndarray, wav2: np.ndarray, window: np.ndarray, window_meta,
                 batch_size=32, wave_chan=1, eeg_chan=64, eeg_len=128, wave_len=128, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.eeg = eeg
        self.wav1 = wav1
        self.wav2 = wav2
        self.window = window
        self.wm = window_meta
        self.wave_chan = wave_chan
        self.eeg_chan = eeg_chan
        self.eeg_len = eeg_len
        self.wave_len = wave_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.floor(self.window.shape[0] / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        window = self.window[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        x, label = self.__data_generation(window)

        return x, label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.window)

    def __data_generation(self, window):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # wav1 = np.empty((self.batch_size, self.wave_chan, self.wave_len))
        # wav2 = np.empty((self.batch_size, self.wave_chan, self.wave_len))
        eeg = np.empty((self.batch_size, self.eeg_chan, self.eeg_len))
        label = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i in range(window.shape[0]):
            start, end = self.window[i, self.wm.start], self.window[i, self.wm.end]
            if self.eeg_len != self.wave_len:
                fs_start, fs_end = math.floor(start / self.eeg_len * self.wave_len), math.floor(
                    end / self.eeg_len * self.wave_len)
            else:
                fs_start, fs_end = start, end
            # wav1[i,] = self.wav1[:, fs_start:fs_end]
            # wav2[i,] = self.wav2[:, fs_start:fs_end]
            eeg[i,] = self.eeg[:, start:end]
            label[i] = self.window[i, self.wm.target] - 1

        return eeg, keras.utils.to_categorical(label, num_classes=2)
