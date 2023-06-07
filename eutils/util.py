import os
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
import pandas as pd
import json
import random
import math
from dotmap import DotMap
from tqdm import tqdm, trange
import re


def makePath(path):
    if not os.path.isdir(path):  # 如果路径不存在
        os.makedirs(path)
    return path


def __get_figsize(x, y):
    min = 1
    max = 30
    scale = 2
    figsize = None
    if min <= x/scale <= max and min <= y/scale <= max:
        figsize = [math.ceil(x/scale), math.ceil(y/10)]
    if x/scale > max or y/scale > max:
        max_size = x if x>y else y
        figsize = [x/max_size * max, y/max_size * max]
    if x/scale < min or y/scale < min:
        min_size = x if x<y else y
        figsize = [x/min_size * min, y/min_size * min]
    return figsize


# 输入_ 具有data（tensor或numpy）、figsize、title、x_label、y_label、fontsize、need_save
def heatmap(_, log_path=None, window_index=None, epoch=None, name=None, label=None):
    import torch
    a = _.data
    if (torch.is_tensor(a)):
        a = a.cpu().detach().numpy()
    a = np.squeeze(a)
    if len(a.shape) == 1:
        a = a[None, :]
    a = normalization(a)
    figsize = __get_figsize(a.shape[1], a.shape[0]) if "figsize" not in _ else _.figsize
    title = "title" if "title" not in _ else _.title
    need_save = "need_save" in _ and _.need_save
    fontsize = 5 if "fontsize" not in _ else _.fontsize

    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(pd.DataFrame(np.round(a, 2)), xticklabels=False, yticklabels=False, square=True, cbar=True)
    if "title" in _ and _.title_visual is True:
        ax.set_title(title, fontsize=fontsize)
    if "x_label" in _:
        ax.set_ylabel(_.y_label, fontsize=fontsize)
    if "y_label" in _:
        ax.set_xlabel(_.x_label, fontsize=fontsize)

    # plt.show()
    if log_path is not None:
        save_path = makePath(f"{log_path}/{name}")
        fig.savefig(f"{save_path}/{window_index}_{title}_{label}.png", format='png', transparent=True)

        if need_save:
            pd.DataFrame(a).to_csv(f"{save_path}/{window_index}_{title}_{label}.csv", header=False, index=False)
    plt.close(fig)


# 将数据写入新文件
def result_logging(file_path, name, data):
    data = pd.DataFrame(data, columns=["start", "end", "target", "index", "trail_number", "subject_number", "reversed", "real", "predict", "classification"])
    data.to_csv(f"{file_path}/result_{name}.csv", index=False)


def heatmap_array(data, figsize=None):
    data = data.cpu().detach().numpy()
    data = np.squeeze(data)
    if len(data.shape) == 1:
        data = data[None, :]
    figsize = (math.floor(data.shape[1] / 5), math.floor(data.shape[0] / 5)) if figsize is None else figsize
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(pd.DataFrame(np.round(data, 2)), xticklabels=True, yticklabels=True, square=True)

    plt.show()
    plt.close(fig)


# spike data: array(10, 32, 15, 15); 10是step，32是channel，15*15是长高
def snn_heatmap(_, log_path=None, window_index=None, epoch=None):
    """
    snn的可视化 'Visual Explanations from Spiking Neural Networks using Interspike Intervals'
    @param _: Dotmap结构 其中包含: data(numpy)、figsize、title、x_label、y_label、fontsize
    @param log_path: 保存图片的路径
    @param window_index: 数据的原始index
    @param epoch: 目前模型的迭代数
    """
    spike = _.data
    __title = _.title

    if len(spike.shape) < 4:
        print(spike.shape)
        print(spike)
        print("error! check the code! ")
    else:
        # SAM实现
        gamma = 0.2
        __time, __channel, __height, __width = spike.shape

        # for t in range(__time):
        #     for c in range(__channel):
        #         if np.sum(spike[t, c]) > 0:
        #             print("[" + str(t) + ", " + str(c) + "] have spike!")

        for time_step in range(1, __time + 1):
            sam_matrix = np.zeros(shape=[__height, __width])
            for h in range(__height):
                for w in range(__width):
                    for c in range(__channel):
                        ncs = 0
                        for t in range(time_step - 1):
                            if spike[t, c, h, w] != 0:
                                ncs += np.exp(-gamma * (time_step - 1 - t))
                        sam_matrix[h, w] += ncs * spike[time_step - 1, c, h, w]

            _.data = sam_matrix
            _.title = __title + "_timestep" + str(time_step)
            heatmap(_, log_path, window_index, epoch)


def read_json(path):
    with open(path, 'r') as load_f:
        json_data = json.load(load_f)
    return DotMap(json_data)


def normalization(data, low=0.0, high=1.0, dim=None):
    """
    把数据归一化到[0, 1]
    @param data: type of ndarray or torch.Tensor
    @param dim: 归一化的维度
    @return: 归一化后的数据
    """
    import torch
    if isinstance(data, np.ndarray):
        _range = np.max(data, axis=dim) - np.min(data, axis=dim)
        return (data - np.min(data, axis=dim)) / _range * (high - low) + low
    elif isinstance(data, torch.Tensor):
        if dim is None:
            _range = torch.max(data) - torch.min(data)
            return (data - torch.min(data)) / _range * (high - low) + low
        else:
            _range = torch.max(data, dim=dim, keepdim=True)[0] - torch.min(data, dim=dim, keepdim=True)[0]
            return (data - torch.min(data, dim=dim, keepdim=True)[0]) / _range * (high - low) + low
    # todo:DataFrame添加全局归一化
    elif isinstance(data, pd.DataFrame):
        _range = data.max() - data.min()
        return (data - data.min()) / _range * (high - low) + low


def standardization(data):
    """
    把数据标准化
    @param data: type of ndarray or torch.Tensor
    @param dim: 归一化的维度
    @return: 归一化后的数据
    """
    if isinstance(data, np.ndarray):
        return (data - np.nanmean(data)) / np.nanstd(data)
    else:
        print("error!")
        exit()


def fft_plot(y: np.ndarray, fs: int):
    from scipy.fftpack import fft
    import matplotlib.pyplot as plt
    if len(y.shape) > 1:
        print("must be 1D array")
        exit()

    length = len(y)
    xf = fs * np.arange(int(len(y) / 2)) / length
    yf = abs(fft(y)) / len(xf)
    yf = yf[:int(len(yf) / 2)]
    plt.plot(xf, yf)
    plt.show()


def get_sub_files(path, find_dir = True):
    temp = None
    for root, dirs, files in os.walk(path, topdown=True):
        temp = dirs if find_dir else files
        break
    return temp


def evaluation(predA, envA, envU, args):
    # Evaluate performance for different analysis window lengths
    blockRange = np.asarray([60, 30, 10, 5, 2, 1, 0.5])  # analysis window lengths

    corrResults = np.zeros((9, len(blockRange)))

    for blockLengthIterator in range(len(blockRange)):
        t = time.time()
        blockLength = int(args.fs * blockRange[blockLengthIterator])
        corrA = np.asarray(range(0, envA.shape[0] - blockLength)) * np.nan
        corrU = np.asarray(range(0, envU.shape[0] - blockLength)) * np.nan

        for block in range(corrA.shape[
                               0]):  # for a specific analysis window, run trough the test set prediction and correlate with attended and unattended envelope
            # print(envA[block:block + blockLength].T.shape)
            # print(predA[block:block + blockLength].T.shape)
            corrA[block] = np.corrcoef(envA[block:block + blockLength].T, predA[block:block + blockLength].T)[0][1]
            corrU[block] = np.corrcoef(envU[block:block + blockLength].T, predA[block:block + blockLength].T)[0][1]

        corrResults[0, blockLengthIterator] = np.nanmean(corrA)
        # corrResults[1, blockLengthIterator] = np.nanstd(corrA)
        corrResults[2, blockLengthIterator] = np.nanmean(corrU)
        # corrResults[3, blockLengthIterator] = np.nanstd(corrU)
        results = np.clip(corrA, -1, 1) > np.clip(corrU, -1, 1)
        # output = np.concatenate([corrA[None, :], corrU[None, :]], axis=0)
        # io.savemat('predict' + str(blockLength) + '.mat', {'results_' + str(blockLength): output})
        corrResults[4, blockLengthIterator] = np.nanmean(
            results)  # Values the networks decision. 1 denotes "correct" zero "wrong". Averages also over the complete test set. This result the networks accuracy!
        accuracy = corrResults[4, blockLengthIterator]
        corrResults[5, blockLengthIterator] = (np.log2(2) + accuracy * np.log2(accuracy) + (1 - accuracy) * np.log2(
            (1 - accuracy + 0.00000001) / 1)) * args.fs / blockLength * 60
        corrResults[6, blockLengthIterator] = blockRange[blockLengthIterator]
        # corrResults[7] = trainPat[0]  # Which participant is evaluated
        # corrResults[8] = startPointTest  # At which time point did the evaluation/test set started

    return corrResults


def monitor(process, multiple, second):
    while True:
        sum = 0
        for ps in process:
            if ps.is_alive():
                sum += 1
        if sum < multiple:
            break
        else:
            time.sleep(second)


def get_gpu_with_max_memory(gpu_list: list):
    if not gpu_list:
        return f"cpu"

    result = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free')
    res = result.read()
    max_memory_gpu = gpu_list[0]
    max_memory = 0
    temp = res.splitlines()
    for i in range(len(temp)):
        if i in gpu_list:
            memory = re.compile(r'[1-9]\d*\.\d*|0\.\d*[1-9]|[1-9]\d*').findall(temp[i])[0]
            if int(memory) > max_memory:
                max_memory_gpu = i
                max_memory = int(memory)

    return f"cuda:{max_memory_gpu}"


def select_device(gpu_list=None):
    import torch
    if torch.cuda.is_available():
        if gpu_list:
            return torch.device('cuda:' + str(get_gpu_with_max_memory(gpu_list)))
        else:
            return torch.device('cuda:' + str(get_gpu_with_max_memory([i for i in range(torch.cuda.device_count())])))
    else:
        return torch.device('cpu')


def get_index_by_fs(origin_fs, new_fs, index):
    return math.floor(index / origin_fs * new_fs)


class WeightConstraint(object):

    def __init__(self, low=0.1, high=1.0):
        self.low = low
        self.high = high

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'vth'):
            w = module.vth.data
            w = w.clamp(self.low, self.high)
            module.vth.data = w
