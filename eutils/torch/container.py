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
from ..container import DecisionWindow, DataMeta, AADModel


# @dataclass
# class TorchModel(AADModel):
#     model: torch.nn.Module
#     loss: torch.nn.Module
#     optim: torch.optim.Optimizer
#     sched: torch.optim.lr_scheduler.ChainedScheduler or None
#     warmup: Any
#     dev: torch.device
#
#     def __init__(self, model, loss, optim, sched=None, warmup=None, dev=torch.device("cpu")):
#         super().__init__()
#         self.model = model.to(dev)
#         self.loss = loss.to(dev)
#         self.optim = optim
#         self.sched = sched
#         self.warmup = warmup
#         self.dev = dev


class EEGDataSet(Dataset):
    def __init__(self, eeg: List[np.ndarray], audio: List[List[np.ndarray]], window: List[DecisionWindow], meta: DataMeta, dev):
        super().__init__()
        self.dev = dev
        self.eeg = eeg
        self.audio = audio
        self.window = window
        self.meta = meta

        self.labels = self._get_labels()
        self.to_tensor()

    def _get_labels(self):
        labels = torch.zeros(len(self.window), dtype=torch.int64)
        for idx in range(len(self.window)):
            labels[idx] = self.window[idx].label
        return labels.to(self.dev)

    def to_tensor(self):
        for trail_idx in range(len(self.eeg)):
            if not isinstance(self.eeg[trail_idx], torch.Tensor):
                if self.eeg[trail_idx].dtype == np.float32:
                    self.eeg[trail_idx] = torch.from_numpy(self.eeg[trail_idx].transpose(-1, -2)).to(self.dev)
                else:
                    self.eeg[trail_idx] = torch.tensor(self.eeg[trail_idx].transpose(-1, -2), dtype=torch.float32, device=self.dev)

        if self.audio is not None:
            for trail_idx in range(len(self.audio)):
                for audio_chan_idx in range(len(self.audio[trail_idx])):
                    pointer = self.audio[trail_idx][audio_chan_idx]
                    if not isinstance(pointer, torch.Tensor):
                        if pointer.dtype == np.float32:
                            self.audio[trail_idx][audio_chan_idx] = torch.from_numpy(pointer.transpose(-1, -2)).to(self.dev)
                        else:
                            self.audio[trail_idx][audio_chan_idx] = torch.tensor(pointer.transpose(-1, -2), dtype=torch.float32, device=self.dev)

    def __getitem__(self, idx):
        trail_idx = self.window[idx].trail_idx
        start = self.window[idx].start
        end = self.window[idx].end

        if self.audio is not None:
            fs_start = math.floor(start / self.meta.eeg_fs * self.meta.wav_fs) if self.meta.eeg_fs != self.meta.wav_fs else start
            wav_len = math.floor((end - start) / self.meta.eeg_fs * self.meta.wav_fs)
            fs_end = fs_start + wav_len
            if fs_end >= self.audio[trail_idx][0].shape[-1] or fs_end >= self.audio[trail_idx][1].shape[-1]:
                fs_end = self.audio[trail_idx][0].shape[-1] - 1
                fs_start = fs_end - wav_len
            wav1 = self.audio[trail_idx][0][:, fs_start:fs_end]
            wav2 = self.audio[trail_idx][1][:, fs_start:fs_end]
        else:
            wav1, wav2 = torch.zeros(1), torch.zeros(1)
        eeg = self.eeg[trail_idx][:, start:end]

        label = self.labels[idx]

        if self.window[idx].reversed is not None:
            if self.window[idx].reversed == 1:
                wav1, wav2 = wav2, wav1
                label = 1 - label

        return wav1, wav2, eeg, label, idx

    def __len__(self):
        return len(self.window)

    def call_epoch(self):
        pass


class CurrSet(Dataset):
    def __init__(self, eeg: torch.Tensor, wav1: torch.Tensor, wav2: torch.Tensor, window: torch.Tensor, window_meta, args, dev):
        super(CurrSet, self).__init__()
        self.args = args
        self.dev = dev
        self.eeg = eeg
        self.wav1 = wav1
        self.wav2 = wav2
        self.window = window
        self.wm = window_meta

        self.to_tensor()
        self.el = self.eeg.shape[-1]
        self.wl = self.wav1.shape[-1] if wav1 is not None else None

    def to_tensor(self):
        if not isinstance(self.eeg, torch.Tensor):
            if self.eeg.dtype == np.float32:
                self.eeg = torch.from_numpy(self.eeg.transpose(-1, -2)).to(self.dev)
            else:
                self.eeg = torch.tensor(self.eeg.transpose(-1, -2), dtype=torch.float32, device=self.dev)

        if self.wav1 is not None and self.wav2 is not None and not isinstance(self.wav1, torch.Tensor):
            if self.eeg.dtype == np.float32:
                self.wav1 = torch.from_numpy(self.wav1.transpose(-1, -2)).to(self.dev)
                self.wav2 = torch.from_numpy(self.wav2.transpose(-1, -2)).to(self.dev)
            else:
                self.wav1 = torch.tensor(self.wav1.transpose(-1, -2), dtype=torch.float32, device=self.dev)
                self.wav2 = torch.tensor(self.wav2.transpose(-1, -2), dtype=torch.float32, device=self.dev)

        if not isinstance(self.window, torch.Tensor):
            if self.window.dtype == np.int64:
                self.window = torch.from_numpy(self.window).to(self.dev)
            else:
                self.window = torch.tensor(self.window, dtype=torch.int64, device=self.dev)

    def __getitem__(self, index):
        start = self.window[index, self.wm.start]
        end = self.window[index, self.wm.end]

        if self.wav1 is not None and self.wav2 is not None:
            fs_start = math.floor(start / self.el * self.wl) if self.el != self.wl else start
            wav_len = math.floor((end - start) / self.el * self.wl)
            fs_end = fs_start + wav_len
            if fs_end >= self.wav1.shape[-1]:
                fs_end = self.wav1.shape[-1] - 1
                fs_start = fs_end - wav_len
            wav1 = self.wav1[:, fs_start:fs_end]
            wav2 = self.wav2[:, fs_start:fs_end]
        else:
            wav1, wav2 = torch.zeros(1), torch.zeros(1)
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



