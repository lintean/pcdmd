import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import mne
from mne.preprocessing import ICA


def load_kul(dataset_name, sub_id):
    # set the parameters
    fs = 128
    trails_num, points_num, channels_num = 8, 360 * fs, 64
    data = np.empty((0, points_num, channels_num))
    label = np.zeros(trails_num)

    # load the data and labels
    data_path = f'D:\\eegdata\\{dataset_name}_origin\\S{sub_id}.mat'
    mat_data = scipy.io.loadmat(data_path)
    channel_names = []
    for k_tra in range(trails_num):
        trail_data = mat_data['trials'][0, k_tra]['RawData'][0, 0]['EegData'][0, 0]
        trail_data = trail_data[:points_num, :]  # drop the data after 50 minutes!
        trail_data = np.expand_dims(trail_data, axis=0)
        data = np.concatenate((data, trail_data), axis=0)

        label[k_tra] = 0 if str(mat_data['trials'][0, k_tra]['attended_ear'][0, 0][0]) == 'L' else 1

        cha_name = []
        for cha in mat_data['trials'][0, k_tra]['FileHeader'][0, 0]['Channels'][0, 0]['Label']:
            cha_name.append(cha[0][0])

        channel_names.append(cha_name)

    return data, label, channel_names, fs


def load_fau(dataset_name, sub_id):
    # set the parameters
    fs = 2500
    trails_num, points_num, channels_num = 6, 285 * fs, 22
    data = np.empty((0, points_num, channels_num))
    label = np.zeros(trails_num)

    # 确定文件编号
    file_name = f'HI{str(sub_id).zfill(3)}.mat' if 1 <= sub_id <= 18 else f'NH{str(sub_id - 14).zfill(3)}.mat'
    data_path = f'../db/AAD_{dataset_name}/{file_name}'

    # load the data and labels
    mat_data = scipy.io.loadmat(data_path)
    channel_names = []
    for k_tra in range(trails_num):
        trail_data = mat_data[f'seg{k_tra + 1}'][0, 0]['eeg_env']
        trail_data = trail_data[:points_num, :]  # drop the data after 285 seconds!
        trail_data = np.expand_dims(trail_data, axis=0)
        data = np.concatenate((data, trail_data), axis=0)

        label[k_tra] = int(mat_data[f'seg{k_tra + 1}'][0, 0]['attention'][0][-1]) - 1

        cha_name = []
        for cha in mat_data[f'seg{k_tra + 1}'][0, 0]['label']:
            cha_name.append(cha[0][0])

        channel_names.append(cha_name)

    return data, label, channel_names, fs


def mne_io_raw(channel_names, dataset_name, k_tra, fs):
    ch_names = channel_names[k_tra]

    # FAU 非标准化通道名称的修订
    if dataset_name == 'FAU':
        replace = {'FP1': 'Fp1', 'FP2': 'Fp2'}
        ch_names = [replace[i] if i in replace else i for i in ch_names]

    # 分配通道类型
    ch_types = list(['eeg' for _ in range(len(ch_names))])

    info = mne.create_info(ch_names, fs, ch_types)

    # 设置电极类型
    montage_dict = {'KUL': 'biosemi64', 'DTU': 'biosemi64', 'FAU': 'standard_1020'}
    info.set_montage(montage_dict[dataset_name])

    return info


def data_loader(dataset_name, sub_id, l_freq, h_freq, is_ica=True):
    """
    加载原始数据库
    :param dataset_name: 数据库名称，如：”KUL“
    :param sub_id: 受试者编号，如：”1“，从1开始！
    :param l_freq: 带通滤波的低频下限
    :param h_freq: 带通滤波的高频上限
    :param is_ica: 是否使用ICA去伪迹
    :return:
        data：脑电数据，trails * time *channels
        label：模型标签， trails * 1
    """
    # load data
    if dataset_name == 'KUL':
        data, label, channel_names, fs = load_kul(dataset_name, sub_id)
    elif dataset_name == 'FAU':
        data, label, channel_names, fs = load_fau(dataset_name, sub_id)

    # 储存处理后的数据
    data_len, channels_num = int(data.shape[1] / fs * 128), data.shape[2]
    data_preprocessed = np.empty((0, data_len, channels_num))

    # TODO: 确定高频数据的滤波方法是否正确（可视化）
    # preprocess the data
    is_verbose = True
    data = np.transpose(data, axes=(0, 2, 1))  # trails*times*channels ==> trails*channels*times
    for k_tra in range(data.shape[0]):
        # convert the data type into raw array
        info = mne_io_raw(channel_names, dataset_name, k_tra, fs)
        raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

        # remove the data by ica
        raw = raw.filter(l_freq=0.5, h_freq=None, verbose=is_verbose)
        raw = raw.resample(256)

        if is_ica:
            ica = ICA(n_components=20, max_iter='auto', random_state=random.randint(0, 100), verbose=is_verbose)
            ica.fit(raw)
            ica.exclude = [0, 1]
            ica.apply(raw)

        # filter the eeg data
        raw = raw.set_eeg_reference(ref_channels='average', verbose=is_verbose)
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=is_verbose)
        raw = raw.resample(128)

        # save the data
        data_tmp = raw.get_data()
        data_tmp = np.transpose(data_tmp, (1, 0))
        data_tmp = np.expand_dims(data_tmp, 0)
        data_preprocessed = np.concatenate([data_preprocessed, data_tmp], axis=0)

        is_verbose = False

    return data_preprocessed, label


if __name__ == '__main__':
    x, y = data_loader('FAU', 1, 1, 50)
    print(x.shape)
    print(y.shape)
