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


def split_data(data, split_percent):
    cell_number = data.shape[0]
    test_window_length = math.floor(cell_number * split_percent)
    start = random.randint(0, cell_number - test_window_length - 1)
    front = data.iloc[:start]
    middle = data.iloc[start:start + test_window_length]
    back = data.iloc[start + test_window_length:]
    train_data = pd.concat([front, back], axis=0, ignore_index=True)
    test_data = middle

    # train_data = data.iloc[:cell_number - test_window_length - 1]
    # test_data = data.iloc[cell_number - test_window_length - 1:]
    return train_data, test_data


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数


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


# split
def get_split_data(data, args):
    if type(data) == type(np.array([])):
        sound = data[:, 0:args.audio_channel]
        sound_not_target = data[:, args.audio_channel:args.audio_channel * 2]
        EEG = data[:, 2 * args.audio_channel:]
    else:
        sound = data.iloc[:, 0:args.audio_channel]
        sound_not_target = data.iloc[:, args.audio_channel:args.audio_channel * 2]
        EEG = data.iloc[:, 2 * args.audio_channel:]
    return sound, EEG, sound_not_target


def add_delay(data, args):
    sound, EEG, sound_not_target = get_split_data(data, args)

    if args.delay >= 0:
        sound = sound.iloc[:sound.shape[0] - args.delay]
        sound_not_target = sound_not_target.iloc[:sound_not_target.shape[0] - args.delay]
        EEG = EEG.iloc[args.delay:, :]
    else:
        sound = sound.iloc[-args.delay:]
        sound_not_target = sound_not_target.iloc[-args.delay:]
        EEG = EEG.iloc[:EEG.shape[0] + args.delay, :]

    sound = sound.reset_index(drop=True)
    sound_not_target = sound_not_target.reset_index(drop=True)
    EEG = EEG.reset_index(drop=True)

    data_pf = pd.concat([sound, sound_not_target, EEG],
                        axis=1, ignore_index=True)
    return data_pf


def read_prepared_data(data_temp, args, local):
    """

    @param data_temp:
    @param args:
    @param local:
    @return: dataframe shape as [total_time, channel]
    """
    data = []

    for l in range(len(args.ConType)):
        sex = pd.read_csv(args.data_document_path + "/csv/" + local.name + args.ConType[l] + ".csv")
        for k in range(args.trail_number):
            # 读取数据
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + local.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)

            # 如果是KUL 先把attend换回来
            if "KUL" in args.data_meta.dataset_name:
                if sex.iloc[k, 0] == 2:
                    temp = Sound_data
                    Sound_data = Sound_data_not_target
                    Sound_data_not_target = temp

            if 0 <= args.isFM <= 1:
                Sound_data, EEG_data, Sound_data_not_target = get_split_data(data_pf, args)
                # 调整左右位置，添加辅助信息
                if args.isDS and sex.iloc[k, args.isFM] == 2:
                    temp = Sound_data
                    Sound_data = Sound_data_not_target
                    Sound_data_not_target = temp
                if "normalization" in args and args.normalization:
                    EEG_data = normalization(EEG_data)
                # 合并
                data_pf = pd.concat(
                    [Sound_data, Sound_data_not_target, EEG_data], axis=1, ignore_index=True)

            # 加入时延
            data_pf = add_delay(data_pf, args)
            data.append(data_pf)

    data = pd.concat(data, axis=0, ignore_index=True)
    return data


# output shape: [(time, feature) (window, feature) (window, feature)]
def window_split(data, args, local):
    random.seed(args.random_seed)
    # 参数初始化
    if "current_flod" not in local:
        if "current_flod" in args and args.cross_validation_fold > 1:
            local.current_flod = args.current_flod
        else:
            local.current_flod = 0
    test_percent = args.test_percent
    cell_number = args.cell_number - abs(args.delay)
    window_lap = args.window_lap if args.window_lap is not None else args.window_length * (1 - args.overlap)
    overlap_distance = max(0, math.floor(1 / (1 - args.overlap)) - 1)
    selection_trails = 0
    if args.isBeyoudTrail:
        selection_trails = random.sample(range(args.trail_number), math.ceil(args.trail_number * test_percent)) if args.cross_validation_fold <= 1 else \
            [i for i in range(math.floor(local.current_flod * args.trail_number / args.cross_validation_fold), math.floor((local.current_flod + 1) * args.trail_number / args.cross_validation_fold))]

    # local.logger.info(selection_trails)
    need_echo = True
    train_set = []
    test_set = []

    # 对于每个ConType进行划分
    for l in range(len(args.ConType)):
        sex = pd.read_csv(args.data_document_path + "/csv/" + local.name + args.ConType[l] + ".csv")

        # 对于ConType里的每个trail进行划分
        for k in range(args.trail_number):
            # 每个trail里的窗口个数
            window_number = math.floor(
                (cell_number - args.window_length) / window_lap) + 1

            # 随机抽取的测试窗口长度
            test_percent = test_percent if args.cross_validation_fold <= 1 else 1 / args.cross_validation_fold
            if args.isBeyoudTrail:
                test_percent = 1 if k in selection_trails else 0
            test_percent = 0 if args.isALLTrain else test_percent
            test_window_length = math.floor(
                (cell_number * test_percent - args.window_length) / window_lap)
            test_window_length = test_window_length if test_percent == 0 else max(
                0, test_window_length)
            test_window_length = test_window_length + 1
            # 随机抽取的测试窗口左右边界
            test_window_left = random.randint(0, window_number - test_window_length) if args.cross_validation_fold <= 1 else \
                math.floor((window_number - test_window_length + 1) * (local.current_flod / args.cross_validation_fold))
            if need_echo:
                local.logger.info(test_window_left)
                need_echo = False
            test_window_right = test_window_left + test_window_length - 1
            target = 1
            if 0 <= args.isFM <= 1:
                target = sex.iloc[k, args.isFM]

            # 对于ConType里的trail里的每个窗口进行划分
            for i in range(window_number):
                left = math.floor(k * cell_number + i * window_lap)
                right = math.floor(left + args.window_length)
                # 如果不是要抽取的测试窗口，即为训练集里的窗口
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    train_set.append(np.array([left, right, target, len(train_set), k, local.subject_number]))
                    if not args.isDS:
                        train_set.append(np.array([right, left, 3 - target, len(train_set), k, local.subject_number]))
                elif test_window_left <= i <= test_window_right:
                    test_set.append(np.array([left, right, target, len(test_set), k, local.subject_number]))
                    if not args.isDS:
                        test_set.append(np.array([right, left, 3 - target, len(test_set), k, local.subject_number]))

    # 重新组织结构
    train_set = np.stack(train_set, axis=0)
    test_set = np.stack(test_set, axis=0) if len(test_set) > 1 else None

    return np.array(data), train_set, test_set


# output shape: [(time, feature) (window, feature) (window, feature)]
def window_split_new(data, args, local):
    data, label = data
    random.seed(args.random_seed)
    # 参数初始化
    if args.cross_validation_fold > 1 and "current_flod" not in local:
        local.current_flod = args.current_flod
    test_percent = args.test_percent
    cell_number = args.cell_number - abs(args.delay)
    window_lap = args.window_lap if args.window_lap is not None else args.window_length * (1 - args.overlap)
    overlap_distance = max(0, math.floor(1 / (1 - args.overlap)) - 1)
    selection_trails = 0
    if args.isBeyoudTrail:
        selection_trails = random.sample(range(args.trail_number), math.ceil(args.trail_number * test_percent)) if args.cross_validation_fold <= 1 else \
            [i for i in range(math.floor(local.current_flod * args.trail_number / args.cross_validation_fold), math.floor((local.current_flod + 1) * args.trail_number / args.cross_validation_fold))]

    # local.logger.info(selection_trails)
    need_echo = True
    train_set = []
    test_set = []

    # 对于每个ConType进行划分
    for l in range(len(args.ConType)):
        # 对于ConType里的每个trail进行划分
        for k in range(args.trail_number):
            # 每个trail里的窗口个数
            window_number = math.floor(
                (cell_number - args.window_length) / window_lap) + 1

            # 随机抽取的测试窗口长度
            test_percent = test_percent if args.cross_validation_fold <= 1 else 1 / args.cross_validation_fold
            if args.isBeyoudTrail:
                test_percent = 1 if k in selection_trails else 0
            test_percent = 0 if args.isALLTrain else test_percent
            test_window_length = math.floor(
                (cell_number * test_percent - args.window_length) / window_lap)
            test_window_length = test_window_length if test_percent == 0 else max(
                0, test_window_length)
            test_window_length = test_window_length + 1
            # 随机抽取的测试窗口左右边界
            test_window_left = random.randint(0, window_number - test_window_length) if args.cross_validation_fold <= 1 else \
                math.floor((window_number - test_window_length + 1) * (local.current_flod / args.cross_validation_fold))
            if need_echo:
                local.logger.info(test_window_left)
                need_echo = False
            test_window_right = test_window_left + test_window_length - 1
            target = label[k] + 1

            # 对于ConType里的trail里的每个窗口进行划分
            for i in range(window_number):
                left = math.floor(k * cell_number + i * window_lap)
                right = math.floor(left + args.window_length)
                # 如果不是要抽取的测试窗口，即为训练集里的窗口
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    train_set.append(np.array([left, right, target, len(train_set), k, local.subject_number], dtype=int))
                    if not args.isDS:
                        train_set.append(np.array([right, left, 3 - target, len(train_set), k, local.subject_number], dtype=int))
                elif test_window_left <= i <= test_window_right:
                    test_set.append(np.array([left, right, target, len(test_set), k, local.subject_number], dtype=int))
                    if not args.isDS:
                        test_set.append(np.array([right, left, 3 - target, len(test_set), k, local.subject_number], dtype=int))

    # 重新组织结构
    train_set = np.stack(train_set, axis=0)
    test_set = np.stack(test_set, axis=0) if len(test_set) > 1 else None

    return np.array(data), train_set, test_set


def change(data_temp, args, local):
    """
    把数据变成夹心式
    @param data_temp: shape as [sound, sound_not_target, EEG]
    @param args: 参数
    @param local: 参数
    @return: shape as [sound, EEG, sound_not_target]
    """
    train_sound, train_EEG, train_sound_not_target = get_split_data(data_temp, args)
    data_pf = pd.concat(
        [train_sound, train_EEG, train_sound_not_target], axis=1, ignore_index=True)
    return data_pf


# train
def vali_split(train, args):
    if args.vali_percent <= 0:
        return train, np.empty(shape=[0, 1])

    window_number = train.shape[0]
    # 随机抽取的验证窗口长度
    vali_window_length = math.floor(window_number * args.vali_percent)
    # 随机抽取的验证窗口
    vali_window_left = random.randint(0, window_number - vali_window_length)
    vali_window_right = vali_window_left + vali_window_length - 1
    # 重复距离
    overlap_distance = math.floor(1 / (1 - args.overlap)) - 1

    train_window = []
    vali_window = []

    for i in range(window_number):
        # 如果不是要抽取的验证窗口
        if vali_window_left - i > overlap_distance or i - vali_window_right > overlap_distance:
            train_window.append(train[i])
        elif i >= vali_window_left and i <= vali_window_right:
            vali_window.append(train[i])

    return np.array(train_window), np.array(vali_window)


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
