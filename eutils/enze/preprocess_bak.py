# python3
# encoding: utf-8
# 
# @Time    : 2021/12/22 14:51
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : preprocess.py
# @Software: Pycharm
import random

import mne
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from device_to_use import origin_data_document

# DTU: 49.99秒，取49秒, 共计60个Trial
# KUL: 389 - 393秒，取389秒

# 基本设置
montage_dict = {'KUL': 'biosemi64', 'DTU': 'biosemi64', 'SCUT': 'standard_1020'}
fs_dict = {'KUL': 128, 'DTU': 512, 'SCUT': 1000}
ica_kul = [[0, 2], [0, 2], [0, 2], [1, 5], [0, 2], [2, 3], [1, 9], [0, 8], [0, 3], [0, 2], [0, 2], [0, 5], [0, 2],
           [0, 2], [0, 5], [1, 4]]
ica_scut = [[0], [0, 1], [0, 1], [1, 4]]

channel_names_scut = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                      'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10',
                      'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4',
                      'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
                      'FT9', 'FT10', 'Fpz', 'CPz', 'FCz']


def preprocess(dataset_name, sub_id, l_freq=1, h_freq=50, is_ica=True, time_len=1):
    """
    读取数据库的数据，经过ICA去伪迹，然后带通滤波，最后输出标准的样本及标签
    Args:
        dataset_name: 数据库名称
        sub_id: 需要提取的受试者编号
        l_freq: 带通滤波器参数，低频下限
        h_freq: 带通滤波器参数，高通上限
        is_ica: 是否进行ICA处理
        time_len:样本的时间长度（秒）

    Returns:
        data：原始数据，N*P*C，N为样本数量，P为样本长度（帧数），C为通道数量（64）采样率降低为128Hz
        label：数据标签：N*2

    """

    # 加载数据，data：Trail*Time*Channels，label：Trail*1
    data, label = data_loader(dataset_name, sub_id)

    # ICA预处理
    data = data_ica(data, dataset_name, sub_id) if is_ica else data

    # 滤波过程， 采样率降低为128Hz
    data = data_filter(data, dataset_name, l_freq, h_freq)

    # 数据标准化，data：N*Time*Channel，label：N*2
    # data, label = data_split(data, label, time_len)

    return data, label


def data_loader(dataset_name, sub_id):
    """
    加载指定数据库的数据，输出原始频率。
    Args:
        dataset_name: 数据库名称
        sub_id: 需要提取的受试者编号
    Returns:
        data:Trail*Time*Channel
        label:Trail*1

    """
    data, label = [], []

    # 建立数据存储空间
    if dataset_name == 'KUL':
        # 基本信息
        fs = 128
        trails_num, points_num, channels_num = 8, 389 * fs, 64

        # 准备最终储存数据的变量
        data = np.empty((0, points_num, channels_num))
        label = np.zeros((0, 1))

        # 加载数据
        data_path = f'{origin_data_document}\\KUL_origin\\S{sub_id}.mat'
        data_mat = scipy.io.loadmat(data_path)
        for k_tra in range(trails_num):
            print(f'data loader, trail: {k_tra}')
            trail_data = data_mat['trials'][0, k_tra]['RawData'][0, 0]['EegData'][0, 0]
            trail_data = trail_data[:points_num, :]
            trail_data = np.expand_dims(trail_data, 0)

            trail_label = str(data_mat['trials'][0, k_tra]['attended_ear'][0, 0][0])
            trail_label = 0 if trail_label == 'L' else 1
            trail_label = np.expand_dims(np.expand_dims(np.array(trail_label), 0), 0)

            data = np.concatenate((data, trail_data), axis=0)
            label = np.concatenate((label, trail_label), axis=0)
        label = np.squeeze(label, -1)
    elif dataset_name == 'DTU':
        # 基本信息
        fs = 512
        trails_num, points_num, channels_num = 20, 49 * fs, 72

        # 准备最终储存数据的变量
        data = np.empty((0, points_num, channels_num))
        label = np.zeros((0, 1))

        # 加载数据
        data_path = f'{origin_data_document}\\DTU_origin\\S{sub_id}.mat'
        data_mat = scipy.io.loadmat(data_path)

        # 划分不同的Trials
        data_all_trails = data_mat['data'][0, 0]['eeg'][0, 0][:, 0:72]
        event_index = data_mat['data'][0, 0]['event'][0, 0]['eeg'][0, 0]['sample']
        n_speakers = data_mat['data'][0, 0]['n_speakers']
        acoustic_condition = data_mat['data'][0, 0]['acoustic_condition']

        # # 利用乳突电极进行基线校正
        # data_mat = data_mat - np.expand_dims(np.mean(data_mat[:, 64:66], axis=-1), -1)

        # 加载数据
        for k_tra in range(len(n_speakers)):
            print(f'data loader, trail: {k_tra}')

            # 1 的时候，是指准备时间。
            if n_speakers[k_tra] == 2 and acoustic_condition[k_tra] == 1:
                ind_s, ind_e = event_index[2 * k_tra, 0], event_index[2 * k_tra + 1, 0]
                trail_data = data_all_trails[ind_s:ind_e, :]
                trail_data = trail_data[:points_num, :]
                trail_data = np.expand_dims(trail_data, 0)

                trail_label = data_mat['data'][0, 0]['attend_lr'][k_tra, 0]
                trail_label = np.expand_dims(np.expand_dims(trail_label, 0), 0)

                data = np.concatenate((data, trail_data), axis=0)
                label = np.concatenate((label, trail_label), axis=0)

        label = np.squeeze(label, -1) - 1
    elif dataset_name == 'SCUT':
        # 基本信息
        fs = 1000
        trails_num, points_num, channels_num = 32, 55 * fs, 64

        # 准备最终储存数据的变量
        data = np.empty((0, points_num, channels_num))

        # 加载数据
        # files = dir(data_path, '.mat')
        # TODO: 适配不同的断点数据
        data_path = f'{origin_data_document}\\experiment\\S{sub_id}.mat'
        data_mat = scipy.io.loadmat(data_path)

        # 输入格式化
        for k_tra in range(trails_num):
            k_sta = data_mat['Markers'][0, 3 * k_tra + 2][3][0][0]

            trail_data = np.empty((1, points_num, channels_num))
            for k_cha in range(len(channel_names_scut)):
                trail_data[0, :, k_cha] = data_mat[channel_names_scut[k_cha]][k_sta:k_sta + points_num, 0]

            data = np.concatenate((data, trail_data), axis=0)
        label = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    else:
        print('Error, check the "dataset_name"!')

    return data, label


def data_ica(data, dataset_name, sub_id):
    """
    对数据进行ICA处理，去伪迹，同时去除50Hz工频干扰
    Args:
        data: 原始输入数据
        dataset_name: 数据库的名称
        sub_id: 受试者编号，KUL手动去除ica时要用

    Returns:
        data: 处理后的数据

    """

    # 准备电极信息
    info = set_info(dataset_name)

    # 将数据转化为EpochsArray格式以计算ica
    data = np.transpose(data, (0, 2, 1))
    data = mne.filter.notch_filter(data, Fs=fs_dict[dataset_name], freqs=50)  # 陷波50Hz
    raw_ica = mne.EpochsArray(data, info)

    # 计算ica数据
    raw_ica = raw_ica.filter(l_freq=1, h_freq=None)
    ica = ICA(n_components=20, max_iter='auto', random_state=97)  # 97 不能够改变，决定了手动ICA的标号
    ica.fit(raw_ica)

    # 去眼电
    is_verbose = True
    for k_tra in range(data.shape[0]):
        print(f'data ica, trail: {k_tra}')

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

        # 计算剔除的通道
        if dataset_name == 'KUL':
            ica.exclude = ica_kul[int(sub_id) - 1]
        elif dataset_name == 'SCUT':
            ica.exclude = ica_scut[int(sub_id) - 1]
        elif dataset_name == 'DTU':
            ica.exclude = []
            exclude_list = []
            eog_channels = ['eog1', 'eog2', 'eog3', 'eog4', 'eog5', 'eog6']
            for cha in eog_channels:
                eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=cha, verbose=is_verbose)
                exclude_list += eog_indices
            ica.exclude = list(set(exclude_list))

        # 基于ICA去眼电
        ica.apply(raw, verbose=is_verbose)

        # 储存数据
        data[k_tra] = raw.get_data()

        # 关闭可视化过程
        is_verbose = False

    data = np.transpose(data, (0, 2, 1))

    return data


def data_filter(data, dataset_name, l_freq, h_freq):
    """
    对数据进行滤波处理，并降低采样率到128Hz（标准化的采样率）
    Args:
        data: 去伪迹后的数据
        dataset_name: 数据库名称
        l_freq:带通滤波的低频范围
        h_freq:带通滤波的高频范围

    Returns:
        data: 滤波后的数据

    """

    data = np.transpose(data, (0, 2, 1))

    if dataset_name == 'KUL':
        points_num = data.shape[2]
    elif dataset_name == 'DTU':
        points_num = int(data.shape[2] / 512 * 128)
    elif dataset_name == 'SCUT':
        points_num = int(data.shape[2] / 1000 * 128)

    # 建立空矩阵储存数据
    data_resample = np.empty((0, 64, points_num))

    # 去眼电
    is_verbose = True
    info = set_info(dataset_name)
    for k_tra in range(data.shape[0]):
        print(f'data filter, trail: {k_tra}')

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

        # 重参考、滤波、降采样
        raw = raw.set_eeg_reference(ref_channels='average', verbose=is_verbose)
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=is_verbose)
        raw = raw.resample(128)

        # 储存数据
        trail_data = raw.get_data()[0:64, :]
        trail_data = np.expand_dims(trail_data, 0)
        data_resample = np.concatenate([data_resample, trail_data], axis=0)

        # 关闭可视化过程
        is_verbose = False

    # 转置，变成 Trail*Time*Channel
    data = np.transpose(data_resample, (0, 2, 1))

    return data


def data_split(data, label, time_len):
    """
    对数据进行标准化，变成N*P*C，P为样本长度（帧数），C为通道数量
    Args:
        data:处理后的干净数据（Trails*Times*Channels）
        label:标签
        time_len:样本时长（秒）

    Returns:
        data：格式化数据
        label：格式化标签

    """

    sample_len = int(128 * time_len)

    # cutoff
    trails_num, points_num, channels_num = data.shape
    samples_num = points_num // sample_len
    data = data[:, :samples_num * sample_len, :]
    data = np.reshape(data, [trails_num, samples_num, sample_len, channels_num])
    # reshape the data into N*T*C
    data = np.transpose(data, [1, 0, 2, 3])
    data = np.reshape(data, [-1, sample_len, channels_num])

    # one-hot encoding
    label = np.expand_dims(label, axis=-1) * np.ones((1, samples_num))
    label = np.transpose(label, [1, 0])
    label = np.reshape(label, -1)
    label = np.eye(2)[label.astype(int)]

    return data, label


def set_info(dataset_name):
    """
    设置电极信号（用于mne的数据格式转换）
    Args:
        dataset_name:数据库名称

    Returns:
          info：通道数据等

    """

    if dataset_name == 'SCUT':
        ch_names = channel_names_scut
    else:
        ch_names = mne.channels.make_standard_montage(montage_dict[dataset_name]).ch_names
    ch_types = list(['eeg' for _ in range(len(ch_names))])
    if dataset_name == 'DTU':
        ch_names = ch_names + ['ecg1', 'ecg2', 'eog1', 'eog2', 'eog3', 'eog4', 'eog5', 'eog6']
        ch_types = ch_types + ['ecg', 'ecg', 'eog', 'eog', 'eog', 'eog', 'eog', 'eog']
    info = mne.create_info(ch_names, fs_dict[dataset_name], ch_types)
    info.set_montage(montage_dict[dataset_name])

    return info


if __name__ == '__main__':
    x, y = preprocess('SCUT', '2', 1)
    # for k in range(16):
    #     x, y = preprocess('KUL', str(k + 1))
    print(x.shape)
    print(y.shape)
