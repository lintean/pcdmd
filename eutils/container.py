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
    """数据相关的参数

    Args:
        dataset: AADDataset类型，可为"DTU"、"KUL"、"SCUT"
        subj_id: 被试的ID。如subj_id="1"为被试1
        trail_num: trail的数量。如trail_num=8
        con_type: List[ConType]。需要读取的数据的声学环境。如con_type=["No", "Low", "High"]
        eeg_fs: 数据的eeg采样率。如eeg_fs=128
        wav_fs: 数据的语音采样率。如wav_fs=128

        eeg_band: 数据的eeg频带数。一般情况下eeg为单频带，因此eeg_band=1
        eeg_band_chan: 数据的eeg中 每个频带的通道数。一般情况下通道数为64。因此eeg_band_chan=64
        eeg_chan: 数据的eeg的总通道数。这一项不需要手动设置，会自动计算得出。为eeg_band * eeg_band_chan
        wav_band: 数据的语音频带数。一般情况下语音为单频带，因此eeg_band=1
        wav_band_chan: 数据的语音中 每个频带的通道数。一般情况下通道数为1。因此eeg_band_chan=1
        wav_chan: 数据的语音的总通道数。这一项不需要手动设置，会自动计算得出。为wav_band * wav_band_chan
        chan_num: 数据的eeg和语音的总通道数。这一项不需要手动设置，会自动计算得出。为eeg_chan + wav_chan
    """
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
    """窗口数据的容器。

    在划分窗口后，数据被划分为一个个窗口。DecisionWindow代表了一个窗口，但没有存储独立的数据，而是存储了能定位数据的“坐标”。
    通过DecisionWindow内的坐标，能找到该窗口的数据。

    Args:
        trail_idx: 该窗口数据所属trail的下标。如trail_idx=0
        start: 该窗口数据在所属trail内 的起始坐标。如start=0
        end: 该窗口数据在所属trail内 的结束坐标。如end=128
        label: 该窗口数据对应的label。如label=0
        win_idx: 该窗口数据在所属trail内 的下标。因此目前不同trail会出现相同的下标。
        subj_idx: 该窗口数据所属被试的ID。如subj_idx="1"
        reversed: 该窗口数据在训练时是否需要反转。反转指将语音对调并将label取反。特殊训练时使用。
    """
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
    """划分窗口、划分数据集的参数

    Args:
        time_len: 划分窗口的时间长度。单位为秒。如time_len=1
        time_lap: 划分窗口时 相邻窗口起始点的时间间隔。单位为秒。如time_lap=0.5。该属性和overlap只能设置一个
        overlap: 划分窗口的重叠率。如overlap=0.5。该属性和time_lap只能设置一个
        delay: 划分窗口时，给脑电添加的时延。目前该属性已废弃。
        cv_flod: 如果采用交叉验证，所使用的总折数。如cv_flod=5。当训练流程中有cv_divide时，该属性必须被设置。
        curr_flod: 如果采用交叉验证，该次训练第几折。如curr_flod=0。当运行main且训练流程中有cv_divide时该属性必须被设置。当运行multiple_train且训练流程中有cv_divide时，可以不设置该属性使程序自动运行所有折数。
        tes_pct: 如果采用hold on验证，需要设置的测试集百分比。如tes_pct=0.2
        valid_pct: 如果采用hold on验证，需要设置的验证集百分比。如tes_pct=0
    """
    time_len: float = 1
    time_lap: float = None
    overlap: float = None
    delay: int = 0
    cv_flod: int = None
    curr_flod: int = None
    tes_pct: float = 0.2
    valid_pct: float = 0.0

    def __post_init__(self):
        if self.time_lap is None and self.overlap is None:
            raise ValueError("time_lap or overlap in SplitMeta must not be empty")

        if self.overlap is not None:
            self.time_lap = self.time_len * (1 - self.overlap)

    def __iter__(self):
        return self.__dict__


@dataclass
class AADModel:
    """模型相关容器

    Args:
        model: 模型
        loss: 训练使用的loss函数
        optim: 训练使用的优化器
        sched: 训练使用的学习率策略。为None则不采用。
        warmup: 一种特殊的学习率策略。
        dev: 模型和数据放置的device。cpu中或特定gpu中。
    """
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
    """数据相关

    Args:
        meta: 参见DataMeta。数据参数容器
        audio: 语音数据。一般情况下为List[List[np.ndarray]]，第一个list为不同trail，第二个list为trail内不同语音，默认第一个语音是attend
        eeg: eeg数据。一般情况下为List[np.ndarray]。list为不同trail。
        pre_lbl: attend语音的属性。List[Dict]。list为不同trail。Dict标明关注侧的direction和speech。取值从0开始
        labels: 训练用的label。List[int]。list为不同trail。因此AAD实验中同一个trail被试的关注侧一致，因此同一个trail的label也一致。二分类取值为0或1。
        windows: 窗口相关参数。List[List[DecisionWindow]]，第一个list为不同trail，第二个list为trail内不同窗口
        tng_win: 训练集的窗口。List[DecisionWindow]。
        tes_win: 测试集的窗口。List[DecisionWindow]。
        tng_loader: 训练集的DataLoader
        tes_loader: 测试集的DataLoader
        aad_model: 模型相关。参见AADModel
    """
    meta: DataMeta = None
    audio: List[List[np.ndarray]] or np.ndarray = None
    eeg: List[np.ndarray] or np.ndarray = None
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
    """预处理/读取数据的参数

    Args:
        eeg_lf: 预处理中的eeg滤波下限。若训练流程中包含预处理如preproc，该属性必须被设置
        eeg_hf: 预处理中的eeg滤波上限。若训练流程中包含预处理如preproc，该属性必须被设置
        wav_lf: 预处理中的语音滤波下限。若训练流程中包含预处理如preproc，该属性必须被设置
        wav_hf: 预处理中的语音滤波上限。若训练流程中包含预处理如preproc，该属性必须被设置
        label_type: 训练的label类型。在ASAD任务中，需要设置成"direction"使用方位作为label。在有语音的任务中，设置成"direction"意味着添加了方位信息作为先验，设置成"speaker"意味着添加了说话人信息作为先验
        need_voice: 是否需要语音。为了节省资源，在ASAD任务中可以设置为false以加快训练。语音的预处理非常耗资源和时间。
        ica: 预处理中 是否需要ICA。若训练流程中包含预处理如preproc，该属性必须被设置
        internal_fs: 预处理中 语音的中间采样率。按照KUL的处理流程，语音预处理首先进行降采样，该属性指定了降采样后的采样率。若需要语音预处理，该属性必须被设置
        gl: 预处理中 语音的gamma filter bank的滤波下限。若需要语音预处理，该属性必须被设置
        gh: 预处理中 语音的gamma filter bank的滤波上限。若需要语音预处理，该属性必须被设置
        space: 预处理中 语音的gamma filter bank的滤波器组中心间隔。若需要语音预处理，该属性必须被设置
        is_combine: 预处理中 语音的gamma filter bank滤波后语音子带是否需要合并。合并则为单频带，不合并则为多频带。若需要语音预处理，该属性必须被设置
    """
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
    """训练参数，暂未被使用
    """
    batch_size: int = 32
    max_epoch: int = 100
    lr: float = 1e-3
    device: str = "cpu"

    def __iter__(self):
        return self.__dict__


@dataclass
class DataSplit:
    """暂未被使用
    """
    meta: SplitMeta
    split_steps: list


@dataclass
class AADTrain:
    """暂未被使用
    """
    data: AADData
    split: DataSplit
    model: List[AADModel] or AADModel
    meta: TngMeta
    train_steps: list


@dataclass
class AAD:
    """暂未被使用
    """
    label: str
    random_seed: int
    dataset_path: str
    subj_id: str
    train: AADTrain


@dataclass
class MultAAD:
    """暂未被使用
    """
    subjects: List[AAD]



