import numpy as np
import pandas as pd
import random
import math
from dotmap import DotMap

import ecfg
from eutils.util import normalization, standardization
import sys
import scipy.io as scio
from scipy.io import wavfile
from scipy.signal import resample


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


def read_prepared_data(args: DotMap, local: DotMap, **kwargs):
    """
    读取csv中的eeg和语音
    @param data_temp:
    @param args:
    @param local:
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [ConType * trial, time, channel]
    """
    sd, ed, usd, data = [], [], [], []
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
            EEG_data = EEG_data.to_numpy(dtype=np.float32)
            Sound_data = Sound_data.to_numpy(dtype=np.float32)
            Sound_data_not_target = Sound_data_not_target.to_numpy(dtype=np.float32)
            left, right = Sound_data, Sound_data_not_target

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
                    if labels[k] == 2:
                        Sound_data, Sound_data_not_target = Sound_data_not_target, Sound_data
                if "normalization" in args and args.normalization:
                    EEG_data = normalization(EEG_data)

                if "standardization" in args and args.standardization:
                    EEG_data = standardization(EEG_data)
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


def read_cma_data(args: DotMap, local: DotMap, **kwargs):
    """
    读取csv中的eeg和语音
    @param data_temp:
    @param args:
    @param local:
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [ConType * trial, time, channel]
    """
    sd, ed, usd, data = [], [], [], []
    labels_all = []
    dm = scio.loadmat(f'{args.data_document_path}/order.mat')
    left_files, right_files = dm['left'], dm['right']

    for l in range(len(args.ConType)):
        sex = pd.read_csv(args.data_document_path + "/csv/" + local.name + args.ConType[l] + ".csv")
        labels = _get_label(sex, args)
        labels_all.append(labels)
        for k in range(args.trail_number if args.trail_number >= 8 else 8):
            # 读取数据
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + local.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)

            Sound_data, EEG_data, Sound_data_not_target = get_split_data(data_pf, args)
            EEG_data = EEG_data.to_numpy(dtype=np.float32)
            Sound_data = Sound_data.to_numpy(dtype=np.float32)
            Sound_data_not_target = Sound_data_not_target.to_numpy(dtype=np.float32)
            left, right = Sound_data, Sound_data_not_target

            # 筛选音频
            left_file = left_files[local.name_index, k][0][:-4]
            right_file = right_files[local.name_index, k][0][:-4]
            if "cont_judge" in args and args.cont_judge(args, left_file, right_file):
                continue

            left = np.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0)
            right = np.nan_to_num(right, nan=0.0, posinf=0.0, neginf=0.0)
            left = normalization(left)
            right = normalization(right)

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
                Sound_data, Sound_data_not_target = _get_wavab(Sound_data, Sound_data_not_target, labels, k, args)

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


def read_eeg_wav(args: DotMap, local: DotMap, **kwargs):
    """
    读取csv中的eeg和语音
    @param data_temp:
    @param args:
    @param local:
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [ConType * trial, time, channel]
    """
    sd, ed, usd, data = [], [], [], []
    dm = scio.loadmat(f'{args.data_document_path}/order.mat')
    left_files, right_files = dm['left'], dm['right']
    labels_all = []

    for l in range(len(args.ConType)):
        sex = pd.read_csv(args.data_document_path + "/csv/" + local.name + args.ConType[l] + ".csv")
        labels = _get_label(sex, args)
        labels_all.append(labels)
        for k in range(args.trail_number if args.trail_number >= 8 else 8):
            # 读取数据
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + local.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)

            EEG_data = data_pf
            EEG_data = EEG_data.to_numpy()
            audio = pd.read_csv(f'{args.data_document_path}/{args.ConType[l]}/{local.name}Tra{k + 1}_audio.csv',
                                header=None).to_numpy()
            left = audio[:, 0:args.audio_channel]
            right = audio[:, args.audio_channel:args.audio_channel * 2]

            # 额外读取音频
            left_file = left_files[local.name_index, k][0][:-4]
            right_file = right_files[local.name_index, k][0][:-4]

            if "cont_judge" in args and args.cont_judge(args, left_file, right_file):
                continue

            left = np.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0)
            right = np.nan_to_num(right, nan=0.0, posinf=0.0, neginf=0.0)
            left = normalization(left)
            right = normalization(right)

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
                Sound_data, Sound_data_not_target = _get_wavab(Sound_data, Sound_data_not_target, labels, k, args)

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


def read_extra_audio(args: DotMap, local: DotMap, **kwargs):
    """
    读取csv中的eeg和语音
    @param data_temp:
    @param args:
    @param local:
    @return: (wav1, wav2, EEG), args, local; wav1 wav2 EEG shape as [ConType * trial, time, channel]
    """
    np.random.seed(args.random_seed)
    sd, ed, usd, data = [], [], [], []
    dm = scio.loadmat(f'{args.data_document_path}/order.mat')
    left_files, right_files = dm['left'], dm['right']
    labels_all = []

    for l in range(len(args.ConType)):
        sex = pd.read_csv(f"{args.data_document_path}/csv/{local.name}{args.ConType[l]}.csv")
        labels = _get_label(sex, args)
        labels_all.append(labels)
        for k in range(args.trail_number if args.trail_number >= 8 else 8):
            # 读取数据
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + local.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)

            EEG_data = data_pf
            if EEG_data.shape[-1] > 64:
                __s1, EEG_data, __s2 = get_split_data(data_pf, args)
                EEG_data = EEG_data.to_numpy(dtype=np.float32)
            else:
                EEG_data = EEG_data.to_numpy()

            # 额外读取音频
            left_file = left_files[local.name_index, k][0][:-4]
            right_file = right_files[local.name_index, k][0][:-4]
            if "cont_judge" in args and args.cont_judge(args, left_file, right_file):
                continue
            if "hrtf_transform" in args and args.hrtf_transform:
                left_file = left_file.replace("hrtf", "dry")
                right_file = right_file.replace("hrtf", "dry")
            audio_path = f"{ecfg.origin_data_document}/KUL_origin/stimuli_{args.extra_audio}"
            fs, left = wavfile.read(f'{audio_path}/{left_file}.wav')
            fs, right = wavfile.read(f'{audio_path}/{right_file}.wav')

            if "audio_fs" in args and fs > args.audio_fs:
                left = resample(left, math.floor(len(left) / fs * args.audio_fs))
                right = resample(right, math.floor(len(right) / fs * args.audio_fs))
                fs = args.audio_fs

            left = left[..., None]
            right = right[..., None]
            left = left[:math.floor(EEG_data.shape[0] / args.fs * fs), :]
            right = right[:math.floor(EEG_data.shape[0] / args.fs * fs), :]

            left = np.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0)
            right = np.nan_to_num(right, nan=0.0, posinf=0.0, neginf=0.0)
            left = normalization(left)
            right = normalization(right)

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
                Sound_data, Sound_data_not_target = _get_wavab(Sound_data, Sound_data_not_target, labels, k, args)
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
            test_window_length = math.floor(
                (cell_number * test_percent - args.window_length) / window_lap)
            test_window_length = max(0, test_window_length) + 1

            # 随机抽取的测试窗口左右边界
            test_window_left = random.randint(0,
                                              window_number - test_window_length) if args.cross_validation_fold <= 1 else \
                math.floor((window_number - test_window_length + 1) * (local.current_flod / args.cross_validation_fold))
            test_window_right = test_window_left + test_window_length - 1
            target = 1
            if 0 <= args.isFM <= 1:
                target = labels[k]

            # 对于ConType里的trail里的每个窗口进行划分
            for i in range(window_number):
                left = math.floor(k * cell_number + i * window_lap)
                right = math.floor(left + args.window_length)
                # 如果不是要抽取的测试窗口，即为训练集里的窗口
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    train_set.append(np.array([left, right, target, len(train_set), k, local.subject_number]))
                elif test_window_left <= i <= test_window_right:
                    test_set.append(np.array([left, right, target, len(test_set), k, local.subject_number]))

    # 重新组织结构
    train_set = np.stack(train_set, axis=0)
    test_set = np.stack(test_set, axis=0) if len(test_set) > 1 else None

    if len(data[-1].shape) > 2:
        wav1 = data[0].reshape([-1, args.audio_channel]) if data[0] is not None else None
        wav2 = data[1].reshape([-1, args.audio_channel]) if data[1] is not None else None
        eeg = data[2].reshape([-1, args.eeg_channel]) if data[2] is not None else None
        data = (wav1, wav2, eeg)
    return (data, train_set, test_set), args, local


def get_reverse(length):
    up, down = length // 2 + length % 2, length // 2
    up, down = np.zeros(up, dtype=np.int), np.zeros(down, dtype=np.int) + 1
    temp = np.concatenate([up, down])
    np.random.shuffle(temp)
    return temp


def add_reverse(data, args: DotMap, local: DotMap):
    data, train_window, test_window = data
    reverse = get_reverse(train_window.shape[0])[..., None]
    train_window = np.concatenate([train_window, reverse], axis=1)
    reverse = get_reverse(test_window.shape[0])[..., None]
    test_window = np.concatenate([test_window, reverse], axis=1)
    return (data, train_window, test_window), args, local


def copy_reverse_window(data, args: DotMap, local: DotMap):
    def __copy_reverse_window(window):
        pass

    data, train_window, test_window = data
    train_window = __copy_reverse_window(train_window)
    test_window = __copy_reverse_window(test_window)
    return (data, train_window, test_window), args, local


def remove_repeated(data, args: DotMap, local):
    data, train_window, test_window = data

    delete_axis = []
    for i in range(test_window.shape[0]):
        for j in range(train_window.shape[0]):
            if train_window[j, 0] < test_window[i, 0] < train_window[j, 1] or train_window[j, 0] < test_window[i, 1] < \
                    train_window[j, 1]:
                delete_axis.append(j)
    local.logger.info(f"remove repeated training window: {delete_axis}")
    train_window = np.delete(train_window, delete_axis, axis=0)

    return (data, train_window, test_window), args, local


def add_negative_samples(data, args: DotMap, local):
    def __add_negative_samples(window):
        negative_window = np.array(window)
        window = np.concatenate([window, np.zeros(shape=(window.shape[0], 1))], axis=1)
        negative_window = np.concatenate([negative_window, np.ones(shape=(negative_window.shape[0], 1))], axis=1)
        # 给window index一个bias
        negative_window[:, 3] += window.shape[0]
        return np.concatenate([window, negative_window], axis=0)

    data, train_window, test_window = data
    new_train_window = __add_negative_samples(train_window)
    new_test_window = __add_negative_samples(test_window)
    local.logger.info(f"add negative samples to training window: from {train_window.shape} to {new_train_window.shape}")
    local.logger.info(f"add negative samples to test window: from {test_window.shape} to {new_test_window.shape}")
    return (data, new_train_window, new_test_window), args, local


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


def preprocess_eeg(args: DotMap, local: DotMap, **kwargs):
    """
    从恩泽师兄开发的预处理程序中读取EEG以及相关的meta
    目前只读取EEG 不读取语音
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
            data_temp = preprocess(dataset, local.name[1:], l_freq=1, h_freq=32)
            data_temp = (data_temp[:, :, 0:1], data_temp[:, :, 0:1], data_temp)
            return data_temp, args, local


def preprocess_eeg_audio(args: DotMap, local: DotMap, **kwargs):
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
        if dataset in args.data_meta.dataset_name:
            ica = True if "ica" not in args else args.ica
            eeg, audio, labels = preprocess(dataset, local.name[1:], l_freq=1, h_freq=32, is_ica=ica,
                                            label_type=args.label_type, need_voice=args.need_voice)
            labels = np.array(labels) + 1
            labels = labels.tolist()
            local.labels = labels
            eeg = np.stack(eeg, axis=0)
            if args.need_voice:
                tmp = np.array(audio)
                output = (tmp[:, 0, ...][..., None], tmp[:, 1, ...][..., None], eeg)
            else:
                output = (None, None, eeg)
            return output, args, local


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


def _get_label(sex, args) -> np.ndarray:
    """
    根据先验信息表（attend的方向和说话人）得到label
    @param sex:
    @param args:
    @return:
    """
    if "prior" not in args:
        raise ValueError("prior is not in args")

    if args.prior == "none":
        # todo: 这里可以改进一下
        while True:
            labels = np.random.randint(1, 3, [args.trail_number])
            if np.count_nonzero(labels - 1) == args.trail_number / 2: break
    elif args.prior == "default":
        labels = np.ones(args.trail_number)
    elif args.prior == "speaker" or args.prior == "direction":
        labels = sex.iloc[:, args.isFM].to_numpy()
    else:
        raise ValueError("prior must be the subject of [default none speaker direction]")
    return labels


def _get_wavab(wav1, wav2, labels, k, args):
    if "prior" not in args:
        raise ValueError("prior is not in args")

    if args.prior == "none":
        if labels[k] == 2:
            return wav2, wav1
    elif args.prior == "default":
        return wav1, wav2
    elif args.prior == "speaker" or args.prior == "direction":
        if labels[k] == 2:
            return wav2, wav1
    else:
        raise ValueError("prior must be the subject of [default none speaker direction]")