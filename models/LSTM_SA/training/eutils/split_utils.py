import numpy as np
import pandas as pd
import random
import math
from dotmap import DotMap
from tqdm import tqdm, trange
from eutils.util import normalization
import sys
import scipy.io as scio


# split
def get_split_data(data, args):
    if isinstance(data, pd.DataFrame):
        sound = data.iloc[:, 0:args.audio_channel]
        sound_not_target = data.iloc[:, args.audio_channel:args.audio_channel * 2]
        EEG = data.iloc[:, 2 * args.audio_channel:]
        return sound, EEG, sound_not_target
    else:
        sys.exit()


def add_delay(data, args):
    sound, sound_not_target, EEG = data

    if args.delay >= 0:
        sound = sound[:sound.shape[0] - args.delay, :]
        sound_not_target = sound_not_target[:sound_not_target.shape[0] - args.delay, :]
        EEG = EEG[args.delay:, :]
    else:
        sound = sound[-args.delay:, :]
        sound_not_target = sound_not_target[-args.delay:, :]
        EEG = EEG[:EEG.shape[0] + args.delay, :]

    return (sound, sound_not_target, EEG)


def read_prepared_data_old(args: DotMap, local: DotMap, **kwargs):
    """

    @param data_temp:
    @param args:
    @param local:
    @return: tuple shape as (sound, unsound, eeg)
    """
    sd, ed, usd, data = [], [], [], []

    for l in range(len(args.ConType)):
        sex = pd.read_csv(args.data_document_path + "/csv/" + local.name + args.ConType[l] + ".csv")
        for k in range(args.trail_number):
            # 读取数据
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + local.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)

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
                data_pf = (Sound_data.to_numpy(), Sound_data_not_target.to_numpy(), EEG_data.to_numpy())

            # 加入时延
            st, ust, et = add_delay(data_pf, args)
            sd.append(st)
            usd.append(ust)
            ed.append(et)

    data = (np.concatenate(sd, axis=0), np.concatenate(usd, axis=0), np.concatenate(ed, axis=0))

    return data


def read_prepared_data(args: DotMap, local: DotMap, **kwargs):
    """
    读取csv中的eeg和语音
    @param data_temp:
    @param args:
    @param local:
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [ConType * trial, time, channel]
    """
    sd, ed, usd, data = [], [], [], []
    # dm = scio.loadmat(f'{args.data_document_path}/order.mat')
    # left_files, right_files = dm['left'], dm['right']
    labels_all = []

    for l in range(len(args.ConType)):
        sex = pd.read_csv(args.data_document_path + "/csv/" + local.name + args.ConType[l] + ".csv")
        labels = sex.iloc[:, args.isFM].to_numpy()
        if "prior" in args and args.prior == "none":
            while True:
                labels = np.random.randint(1, 3, [args.trail_number])
                if np.count_nonzero(labels - 1) == args.trail_number / 2: break;
        labels_all.append(labels)
        for k in range(args.trail_number if args.trail_number >= 8 else 8):
            # 读取数据
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + local.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)

            Sound_data, EEG_data, Sound_data_not_target = get_split_data(data_pf, args)
            # EEG_data = data_pf
            EEG_data = EEG_data.to_numpy()
            Sound_data = Sound_data.to_numpy()
            Sound_data_not_target = Sound_data_not_target.to_numpy()
            left, right = Sound_data, Sound_data_not_target

            # 额外读取音频
            # left_file = left_files[local.name_index, k][0][:-4]
            # right_file = right_files[local.name_index, k][0][:-4]
            # if "hrtf" in left_file: continue
            # left = np.load(f'{args.data_document_path}/150_result/KUL_Separated_TCN_mask/{left_file}.npy')
            # right = np.load(f'{args.data_document_path}/150_result/KUL_Separated_TCN_mask/{right_file}.npy')
            # audio = pd.read_csv(f'{args.data_document_path}/{args.ConType[l]}/{local.name}Tra{k + 1}_audio.csv', header=None).to_numpy()
            # left = audio[:, 0:args.audio_channel]
            # right = audio[:, args.audio_channel:args.audio_channel * 2]
            # left = np.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0)
            # right = np.nan_to_num(right, nan=0.0, posinf=0.0, neginf=0.0)
            # left = normalization(left)
            # right = normalization(right)
            # left = np.swapaxes(left, 0, 1)
            # right = np.swapaxes(right, 0, 1)
            # left = left[:math.floor(EEG_data.shape[0] / args.fs * 800), :]
            # right = right[:math.floor(EEG_data.shape[0] / args.fs * 800), :]

            if len(left.shape) == 1:
                left = left[:, None]
            if len(right.shape) == 1:
                right = right[:, None]

            if 0 <= args.isFM <= 1:
                Sound_data, Sound_data_not_target = left, right

                # 如果是KUL 先把attend换回来
                if "KUL" in args.data_meta.dataset_name:
                    if sex.iloc[k, 0] == 2:
                        Sound_data, Sound_data_not_target = Sound_data_not_target, Sound_data

                # 调整左右位置，添加辅助信息
                if "prior" in args and args.prior == "none":
                    if labels[len(sd)] == 2:
                        Sound_data, Sound_data_not_target = Sound_data_not_target, Sound_data
                else:
                    if args.isDS and labels[k] == 2:
                        Sound_data, Sound_data_not_target = Sound_data_not_target, Sound_data
                if "normalization" in args and args.normalization:
                    EEG_data = normalization(EEG_data)
                # 合并
                data_pf = (Sound_data, Sound_data_not_target, EEG_data)

            # 加入时延
            st, ust, et = add_delay(data_pf, args)

            sd.append(st)
            usd.append(ust)
            ed.append(et)

    # todo: lables没有充分考虑声学环境
    local.labels = labels_all
    data = (np.stack(sd, axis=0), np.stack(usd, axis=0), np.stack(ed, axis=0))

    return data, args, local


def subject_split(data, args: DotMap, local):
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
    window_lap = math.floor(window_lap)
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
        labels = local.labels[l]

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
                target = labels[k]

            # 对于ConType里的trail里的每个窗口进行划分
            for i in range(window_number):
                left = math.floor(k * cell_number + i * window_lap)
                # ?
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

    if len(data[0].shape) > 2:
        wav1 = data[0].reshape([-1, args.audio_channel])
        wav2 = data[1].reshape([-1, args.audio_channel])
        eeg = data[2].reshape([-1, args.eeg_channel])
        data = (wav1, wav2, eeg)
    return (data, train_set, test_set), args, local


def window_split_new(data, args: DotMap, local: DotMap):
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


def vali_split(train, args: DotMap):
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


def get_data_from_preprocess(args: DotMap, local: DotMap, **kwargs):
    """
    从恩泽师兄开发的预处理程序中读取EEG、语音以及相关的meta
    目前只读取EEG
    @param data: 占位符
    @param args: 全局meta
    @param local: subject个人meta
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [ConType * trial, time, channel]
    """
    from eutils.preproc.preprocess import preprocess

    labels_all = []

    for l in range(len(args.ConType)):
        sex = pd.read_csv(args.data_document_path + "/csv/" + local.name + args.ConType[l] + ".csv")
        labels = sex.iloc[:, args.isFM].to_numpy()
        if "prior" in args and args.prior == "none":
            while True:
                labels = np.random.randint(1, 3, [args.trail_number])
                if np.count_nonzero(labels - 1) == args.trail_number / 2: break;
        labels_all.append(labels)
    local.labels = labels_all

    datasets = ['DTU', 'KUL', 'SCUT']
    for dataset in datasets:
        if dataset in args.data_meta.dataset_name:
            data_temp = preprocess('DTU', local.name[1:], l_freq=1, h_freq=32)
            data_temp = (data_temp[:, :, 0:1], data_temp[:, :, 0:1], data_temp)
            return data_temp, args, local


def get_all_data_setter(subject_func):
    """
    从恩泽师兄开发的预处理程序中读取全部人的EEG、语音以及相关的meta
    用于跨被试 目前只读取EEG
    @param data: 占位符
    @param args: 全局meta
    @param local: subject个人meta
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [subject, ConType * trial, time, channel]
    """
    def get_all_data(args: DotMap, local: DotMap, **kwargs):
        _local = DotMap(local)
        wav1, wav2, eeg, labels = [], [], [], []
        for subject in args.names:
            _local.name = subject
            temp, _a, _l = subject_func(args, _local, **kwargs)
            wav1.append(temp[0])
            wav2.append(temp[1])
            eeg.append(temp[2])
            labels.append(_l.labels)
        wav1 = np.stack(wav1, axis=0)
        wav2 = np.stack(wav2, axis=0)
        eeg = np.stack(eeg, axis=0)
        local.labels = labels
        return (wav1, wav2, eeg), args, local

    return get_all_data


def subjects_split(data: tuple, args: DotMap, local: DotMap):
    """
    用于跨被试，将全部人的数据分窗并划分成训练集、测试集
    @param data: tuple(wav1, wav2, EEG); wav1 wav2 EEG shape as [subject, ConType * trial, time, channel]
    @param args: 全局meta
    @param local: subject个人meta
    @return: ((wav1, wav2, EEG), train_window, test_window), args, local;
        wav1 wav2 EEG shape as [subject * ConType * trial * time, channel];
        train_window shape as
    """
    random.seed(args.random_seed)

    train_set = []
    test_set = []
    window_lap = args.window_lap if args.window_lap is not None else args.window_length * (1 - args.overlap)
    window_lap = math.floor(window_lap)
    trail_len = data[-1].shape[2]
    subject_len = data[-1].shape[1] * trail_len

    for subject in range(args.people_number):
        subject_labels = local.labels[subject]
        temp = []
        # 对于每个ConType进行划分
        for l in range(len(args.ConType)):
            labels = subject_labels[l]
            # 对于ConType里的每个trail进行划分
            for k in range(args.trail_number):
                if l == 0 and k == 19:
                    print(subject, l, k)
                left = np.arange(start=0, stop=trail_len, step=window_lap)
                while left[-1] + args.window_length - 1 > trail_len - 1:
                    left = np.delete(left, -1, axis=0)
                left = left + (l * len(args.ConType) + k) * trail_len + subject * subject_len
                right = left + args.window_length
                target = np.ones_like(left) * labels[k]
                window_index = np.arange(start=0, stop=left.shape[0]) + (l * len(args.ConType) + k) * left.shape[0]
                trail_index = np.ones_like(left) * k
                subject_index = np.ones_like(left) * (subject + 1)
                trail_window = [left, right, target, window_index, trail_index, subject_index]

                temp.append(np.stack(trail_window, axis=1))

        if subject + 1 == local.subject_number:
            test_set.append(np.concatenate(temp, axis=0))
        else:
            train_set.append(np.concatenate(temp, axis=0))

    train_set = np.concatenate(train_set, axis=0)
    test_set = np.concatenate(test_set, axis=0)

    wav1 = data[0].reshape([-1, args.audio_channel])
    wav2 = data[1].reshape([-1, args.audio_channel])
    eeg = data[2].reshape([-1, args.eeg_channel])
    return ((wav1, wav2, eeg), train_set, test_set), args, local


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
