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
