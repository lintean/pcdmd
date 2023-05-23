# python3
# encoding: utf-8
#
# @Time    : 2021/12/22 14:51
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : preprocess.py
# @Software: Pycharm
import copy
import os
import mne
import scipy.io
from mne.preprocessing import ICA, corrmap
from db.database import db_path as folder_path
from db.SCUT import scut_eeg_fs, scut_label, channel_names_scut
from .util import *
# DTU: 49.99秒，取49秒, 共计60个Trial
# KUL: 389 - 393秒，取389秒

# 基本设置
montage_dict = {'KUL': 'biosemi64', 'DTU': 'biosemi64', 'SCUT': 'standard_1020'}#some database
fs_dict = {'KUL': 128, 'DTU': 512, 'SCUT': scut_eeg_fs}


#main
def preprocess(dataset_name, sub_id, l_freq=None, h_freq=None, is_ica=True, time_len=1, label_type='direction', window_lap=128):
    """
    读取数据库的数据，经过ICA去伪迹，然后带通滤波，最后输出标准的样本及标签
    Args:
        dataset_name: 数据库名称
        sub_id: 需要提取的受试者编号
        l_freq: 带通滤波器参数，低频下限
        h_freq: 带通滤波器参数，高通上限
        is_ica: 是否进行ICA处理
        time_len:样本的时间长度（秒）
        label_type: 分类标签，方位（direction）或者讲话者（speaker）
        window_lap:

    Returns:
        data：原始数据，N*P*C，N为样本数量，P为样本长度（帧数），C为通道数量（64）
        label：数据标签：N*2

    """
    if h_freq is None:
        h_freq = [50]
    if l_freq is None:
        l_freq = [1]
    if isinstance(l_freq, int):
        l_freq = [l_freq]
        h_freq = [h_freq]

    # 加载数据，data：Trail*Time*Channels，label：Trail*1
    data, label = data_loader(dataset_name, sub_id, label_type)  # 加载的数据一模一样

    # ICA预处理
    data = data_ica(data, dataset_name, label_type) if is_ica else data

    # 滤波过程， 采样率降低为128Hz
    data = data_filter(data, dataset_name, l_freq, h_freq)

    #maybe can add a RMS here

    # 数据标准化，data：N*Time*Channel，label：N*2
    # data, label = data_split(data, label, time_len, len(l_freq), window_lap)
    data, label, split_index = list_data_split(list(data), None, label, time_len, len(l_freq), window_lap)

    return data, label, split_index


def data_loader(dataset_name, sub_id, label_type):
    """
    加载指定数据库的数据，输出原始频率。
    Args:
        dataset_name: 数据库名称
        sub_id: 需要提取的受试者编号
        label_type: 标签的选项
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
        data_path = f'{folder_path}/{dataset_name}_origin/S{sub_id}.mat'
        data_mat = scipy.io.loadmat(data_path)
        for k_tra in range(trails_num):
            print(f'data loader, trail: {k_tra}')
            trail_data = data_mat['trials'][0, k_tra]['RawData'][0, 0]['EegData'][0, 0]
            trail_data = trail_data[:points_num, :]
            trail_data = np.expand_dims(trail_data, 0)

            if label_type == 'direction':
                trail_label = str(data_mat['trials'][0, k_tra]['attended_ear'][0, 0][0])
                trail_label = 0 if trail_label == 'L' else 1
            else:
                trail_label = data_mat['trials'][0, k_tra]['attended_track'][0][0][0][0]
                trail_label = trail_label - 1
            trail_label = np.expand_dims(np.expand_dims(np.array(trail_label), 0), 0)

            data = np.concatenate((data, trail_data), axis=0)
            label = np.concatenate((label, trail_label), axis=0)
        label = np.squeeze(label, -1)
    elif dataset_name == 'DTU':
        # 基本信息
        fs = 512
        trails_num, points_num, channels_num = 20, 49 * fs, 66

        # 准备最终储存数据的变量
        data = np.empty((0, points_num, channels_num))
        label = np.zeros((0, 1))

        # 加载数据
        data_path = f'{folder_path}/AAD_DTU/EEG_new/S{sub_id}.mat'
        data_mat = scipy.io.loadmat(data_path)

        # 划分不同的Trials
        data_all_trails = data_mat['data'][0, 0]['eeg'][0, 0][:, 0:channels_num]
        event_index = data_mat['data'][0, 0]['event'][0, 0]['eeg'][0, 0]['sample']
        n_speakers = data_mat['data'][0, 0]['n_speakers']
        acoustic_condition = data_mat['data'][0, 0]['acoustic_condition']

        # # 利用乳突电极进行基线校正
        # data_all_trails = data_all_trails - np.expand_dims(np.mean(data_all_trails[:, 64:66], axis=-1), -1)

        # 加载数据
        for k_tra in range(len(n_speakers)):
            print(f'data loader, trail: {k_tra}')

            # 1 的时候，是指准备时间。
            if n_speakers[k_tra] == 2 and acoustic_condition[k_tra] == 1:
                ind_s, ind_e = event_index[2 * k_tra, 0], event_index[2 * k_tra + 1, 0]
                trail_data = data_all_trails[ind_s:ind_e, :]
                trail_data = trail_data[:points_num, :]
                trail_data = np.expand_dims(trail_data, 0)

                if label_type == 'direction':
                    trail_label = data_mat['data'][0, 0]['attend_lr'][k_tra, 0]
                else:
                    trail_label = data_mat['data'][0, 0]['attend_mf'][k_tra, 0]
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
        data_path = f'{folder_path}/AAD_{dataset_name}/S{sub_id}/'
        files = os.listdir(data_path)
        files = sorted(files)  # 按顺序，避免label不同
        for file in files:
            # 输入格式化
            data_mat = scipy.io.loadmat(data_path + file)
            for k_tra in range(data_mat['Markers'].shape[1] // 3):
                k_sta = data_mat['Markers'][0, 3 * k_tra + 2][3][0][0]

                trail_data = np.zeros((1, points_num, channels_num))
                # 避免个别通道无效。
                if len(data_mat[channel_names_scut[0]]) >= k_sta + points_num:
                    for k_cha in range(len(channel_names_scut)):
                        trail_data[0, :, k_cha] = data_mat[channel_names_scut[k_cha]][k_sta:k_sta + points_num, 0]

                data = np.concatenate((data, trail_data), axis=0)

        label = copy.deepcopy(scut_label)
        label = scut_order(None, label, sub_id)
        label = select_label(label, label_type=label_type)
        data, label = scut_remove(data, None, label, sub_id)

    else:
        print('Error, check the "dataset_name"!')

    # 保留脑电通道
    data = data[:, :, :]

    return data, label


def data_ica(data, dataset_name, label_type):
    """
    对数据进行ICA处理，去伪迹，同时去除50Hz工频干扰
    Args:
        label_type:
        data: 原始输入数据
        dataset_name: 数据库的名称
    Returns:
        data: 处理后的数据

    """

    # ica_dict = {'KUL': [0, 2], 'DTU': ['eog1', 'eog2', 'eog3', 'eog4', 'eog5', 'eog6'], 'SCUT': [0, 1]}
    ica_dict = {'KUL': [0, 2], 'DTU': [0, 1, 7], 'SCUT': [0, 1]}  # 没有50Hz去工频电的ica
    # ica_dict = {'KUL': [], 'DTU': [0, 5], 'SCUT': []}  # 去掉工频电后的ica

    # 准备电极信息
    info = set_info(dataset_name)

    # 加载模板数据（S1-Trail1）
    data_tmp, label_tmp = data_loader(dataset_name, '1', label_type)
    data_tmp = np.transpose(data_tmp, (0, 2, 1))
    data_tmp = data_tmp[0]

    # 计算ica通道
    raw_tmp = mne.io.RawArray(data_tmp, info)
    raw_tmp = raw_tmp.filter(l_freq=1, h_freq=None)
    raw_tmp.set_montage(montage_dict[dataset_name])
    ica_tmp = ICA(n_components=20, max_iter='auto', random_state=97)
    ica_tmp.fit(raw_tmp)

    # 去眼电
    is_verbose = True
    data = np.transpose(data, (0, 2, 1))
    for k_tra in range(data.shape[0]):
        print(f'data ica, trail: {k_tra}')

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

        # 计算ica数据
        raw = raw.filter(l_freq=1, h_freq=None, verbose=is_verbose)
        ica = ICA(n_components=20, max_iter='auto', random_state=97, verbose=is_verbose)  # 97为随机种子
        ica.fit(raw)

        # 模板匹配法剔除眼电伪迹
        ica_exclude = []
        ica_s = [ica_tmp, ica]
        eog_channels = ica_dict[dataset_name]  # 选取眼电通道
        for k_ica in range(len(eog_channels)):
            corrmap(ica_s, template=(0, eog_channels[k_ica]), threshold=0.9, label=str(k_ica), plot=False,
                    verbose=is_verbose)
            ica_exclude += ica_s[1].labels_[str(k_ica)]

        ica.exclude = list(set(ica_exclude))
        ica.apply(raw, verbose=is_verbose)
        print(ica.exclude)
        del ica
        print(raw)

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

    # 建立空矩阵储存数据
    points_num = int(data.shape[1] / fs_dict[dataset_name] * 128)
    data_resample = np.empty((0, 64, points_num))

    # 滤波
    is_verbose = True
    info = set_info(dataset_name)
    data = np.transpose(data, (0, 2, 1))

    for k_tra in range(data.shape[0]):
        print(f'data filter, trail: {k_tra}')

        # # 重参考、滤波、降采样
        # raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)
        # raw = raw.set_eeg_reference(ref_channels='average', verbose=is_verbose)
        # raw = raw.filter(l_freq=1, h_freq=64, verbose=is_verbose)
        # raw = raw.resample(128)
        # tmp_data = raw.get_data()

        for k in range(len(l_freq)):
            # 将原始数据转化为raw格式文件
            raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

            # 重参考、滤波、降采样
            raw = raw.set_eeg_reference(ref_channels='average', verbose=is_verbose)

            raw = raw.filter(l_freq=l_freq[k], h_freq=h_freq[k], verbose=is_verbose)
            raw = raw.resample(128)

            # 标准化并储存数据
            trail_data = raw.get_data()[0:64, :]
            trail_data = (trail_data - np.average(trail_data, axis=-1)[..., None]) / np.std(trail_data, axis=-1)[..., None]
            trail_data = np.expand_dims(trail_data, 0)
            data_resample = np.concatenate([data_resample, trail_data], axis=0)

            # 关闭可视化过程
            is_verbose = False

    # 转置，变成 Trail*Band*Time*Channel
    data = np.transpose(data_resample, (0, 2, 1))

    return data


def data_split(data, label, time_len, band_num, window_lap):
    """
    对数据进行标准化，变成N*P*C，P为样本长度（帧数），C为通道数量
    Args:
        data:处理后的干净数据（Trails*Times*Channels）
        label:标签
        time_len:样本时长（秒）
        band_num: 频带数量

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

    # reshape the data into N*B*T*C
    data = np.transpose(data, [1, 0, 2, 3])

    trail_num = int(data.shape[1] / band_num)
    data = data.reshape([data.shape[0], trail_num, band_num, sample_len, channels_num])

    data = np.reshape(data, [-1, band_num, sample_len, channels_num])
    data = np.reshape(data, [-1, sample_len, channels_num]) if band_num == 1 else data

    # one-hot encoding
    label = np.expand_dims(label, axis=-1) * np.ones((1, samples_num))
    label = np.transpose(label, [1, 0])
    label = np.reshape(label, -1)
    label = np.eye(2)[label.astype(int)]

    return data, label


def set_info(dataset_name, is_add=False):
    """
    设置电极信号（用于mne的数据格式转换）
    Args:
        dataset_name:数据库名称
        is_add: DTU 是否需要额外的通道数量。

    Returns:
          info：通道数据等

    """

    if dataset_name == 'SCUT':
        ch_names = channel_names_scut
    else:
        ch_names = mne.channels.make_standard_montage(montage_dict[dataset_name]).ch_names
    ch_types = list(['eeg' for _ in range(len(ch_names))])

    if dataset_name == 'DTU' and is_add:
        ch_names = ch_names + ['ecg1', 'ecg2', 'eog1', 'eog2', 'eog3', 'eog4', 'eog5', 'eog6']
        ch_types = ch_types + ['ecg', 'ecg', 'eog', 'eog', 'eog', 'eog', 'eog', 'eog']

    info = mne.create_info(ch_names, fs_dict[dataset_name], ch_types)
    info.set_montage(montage_dict[dataset_name])

    return info


# def run_ica(method, fit_params=None):
#     ica = ICA(n_components=20, method=method, fit_params=fit_params,
#               max_iter='auto', random_state=0)
#     t0 = time()
#     reject = dict(mag=5e-12, grad=4000e-13)
#     ica.fit(raw, reject=reject)
#     fit_time = time() - t0
#     title = ' '
#
#     ica.plot_components(title=title)
#     plt.savefig('D:\\ICA.png')

if __name__ == '__main__':
    x, y = preprocess(dataset_name='SCUT', sub_id='11', l_freq=[1], h_freq=[50.], is_ica=True,
                      time_len=1, label_type='direction')
    # x, y = preprocess(dataset_name='KUL', sub_id='1', l_freq=[1], h_freq=[4], is_ica=True, time_len=1,