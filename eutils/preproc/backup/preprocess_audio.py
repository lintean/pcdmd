# python3
# encoding: utf-8
#
# @Time    : 2021/12/22 14:51
# @Author  : enze
# @Email   : enzesu@hotmail.com
# @File    : preprocess.py
# @Software: Pycharm

# 读取数据，包含脑电信号、音频信号和标签信息
import copy
import os

import mne
import scipy.io
from mne.preprocessing import ICA, corrmap
from scipy import signal
from scipy.io import wavfile
from scipy.signal import hilbert, butter
from db.database import db_path as folder_path
from db.SCUT import scut_eeg_fs, scut_label, trail_number, channel_names_scut
from eutils.preproc.util import *

montage_dict = {'KUL': 'biosemi64', 'DTU': 'biosemi64', 'SCUT': 'standard_1020'}
fs_dict = {'KUL': 128, 'DTU': 512, 'SCUT': scut_eeg_fs}


def preprocess_audio(dataset_name, sub_id, l_freq: int = 1, h_freq=50, is_ica=True, time_len=1, label_type='speaker', window_lap=128, need_voice=True):
    """
    读取数据库的数据，经过ICA去伪迹，然后带通滤波，最后输出标准的样本及标签
    Args:
        dataset_name: 数据库名称
        sub_id: 需要提取的受试者编号
        l_freq: 带通滤波器参数，低频下限
        h_freq: 带通滤波器参数，高通上限
        is_ica: 是否进行ICA处理

    Returns:
        eeg：列表，每个列表包含1个Trail的脑电数据，每个Trail为Time*Channels
        voice：列表，每个列表包含1个Trail的语音数据，每个Trail为Time*2。第一列为左侧音频，第二列为右侧音频
        label：列表，每个列表包含1个Trail的标签，每个标签包含[方位、讲话者]，均以0、1标记。

    """

    print(sub_id)

    # 加载数据
    eeg, voice, label = data_loader(dataset_name, sub_id)  # 加载的数据一模一样

    # 脑电数据预处理（ICA）
    eeg = ica_eeg(eeg, dataset_name, label_type) if is_ica else eeg

    # 滤波过程， 采样率降低为128Hz
    eeg = filter_eeg(eeg, dataset_name, l_freq, h_freq)

    # 语音数据预处理（希尔伯特变换、p-law、滤波）
    voice_fs = 128
    voice = filter_voice(voice, l_freq=l_freq, h_freq=h_freq, fs=voice_fs, is_hilbert=True, is_p_law=True) if need_voice else None

    # label相关处理
    voice = voice_dirct2attd(voice, label)
    voice = voice_attd2speaker(voice, label)
    label = select_label(label, label_type)

    if dataset_name == "DTU":
        labels = []
        for c in range(3):
            temp = []
            for i in range(20):
                temp.append(label[c * 20 + i])
            labels.append(temp)
        label = labels
    else:
        label = [label]
    
    # 数据分窗，data：N*Time*Channel，label：N*2
    # length = len(l_freq) if isinstance(l_freq, list) else 1
    # eeg, voice, label, split_index = list_data_split(eeg, voice, label, time_len, length, window_lap)

    return eeg, voice, label


def data_loader(dataset_name, sub_id):
    """
    读取原始数据。
    Args:
        dataset_name: 数据库名称
        sub_id: 需要提取的受试者编号

    """
    eeg, voice, label = [], [], []

    # 建立数据存储空间
    if dataset_name == 'KUL':
        data_path = f'{folder_path}/AAD_{dataset_name}/S{sub_id}.mat'
        data_mat = scipy.io.loadmat(data_path)

        for k_tra in range(8):
            # 加载语音数据[左侧音频，右侧音频]
            tmp_voice = []
            for k_voice in range(2):
                voice_file = data_mat['trials'][0, k_tra]['stimuli'][0][0][k_voice][0][0]
                voice_path = f'{folder_path}/AAD_{dataset_name}/stimuli/{voice_file}'
                tmp_voice.append(wavfile.read(voice_path)[1])  # 加载语音数据
            # 合并脑电数据
            voice.append(tmp_voice)

            # 加载脑电数据
            tmp_eeg = data_mat['trials'][0, k_tra]['RawData'][0, 0]['EegData'][0, 0]
            eeg.append(tmp_eeg)

            # 加载标签数据[左右，讲话者编号]
            lab = 0 if str(data_mat['trials'][0, k_tra]['attended_ear'][0, 0][0]) == 'L' else 1
            tmp_label = [lab, data_mat['trials'][0, k_tra]['attended_track'][0][0][0][0] - 1]
            label.append(tmp_label)

    elif dataset_name == 'DTU':
        # 加载数据
        data_path = f'{folder_path}/AAD_DTU/EEG_new/S{sub_id}.mat'
        data_mat = scipy.io.loadmat(data_path)

        # 划分不同的Trials
        channels_name = data_mat['data'][0, 0]['dim'][0, 0]['chan'][0, 0]['eeg'][0, 0].squeeze()
        data_all_trails = data_mat['data'][0, 0]['eeg'][0, 0]
        event_index = data_mat['data'][0, 0]['event'][0, 0]['eeg'][0, 0]['sample']
        n_speakers = data_mat['data'][0, 0]['n_speakers']
        male_list = data_mat['data'][0, 0]['wavfile_male']
        female_list = data_mat['data'][0, 0]['wavfile_female']

        chns = []
        for i in range(channels_name.shape[0]):
            chns.append(channels_name[i][0])
        chns = np.stack(chns)

        # 去除EOG通道
        data_all_trails = data_all_trails[:, :66]

        # 计算正式实验的索引
        n_speakers = np.array(n_speakers)
        index_two_speakers = np.where(n_speakers == 2)

        # 加载数据
        for k_tra in index_two_speakers[0]:
            # 读取标签信息
            tmp_label = [data_mat['data'][0, 0]['attend_lr'][k_tra, 0],
                         data_mat['data'][0, 0]['attend_mf'][k_tra, 0]]
            label.append(tmp_label)

            # 加载语音数据
            male_name = male_list[k_tra][0][0]
            female_name = female_list[k_tra][0][0]
            tmp_voice = []
            voice_file_list = [female_name, male_name] if tmp_label[0] + tmp_label[1] == 1 else [male_name, female_name]
            for voice_file_name in voice_file_list:
                voice_path = f'{folder_path}/AAD_{dataset_name}/AUDIO/{voice_file_name}'
                tmp_voice.append(wavfile.read(voice_path)[1])  # 加载语音数据
            voice.append(tmp_voice)

            # 读取脑电数据
            ind_s, ind_e = event_index[2 * k_tra, 0], event_index[2 * k_tra + 1, 0]
            tmp_eeg = data_all_trails[ind_s:ind_e, 0:66]
            eeg.append(tmp_eeg)

    elif dataset_name == 'SCUT':
        # 加载语音数据
        for k_tra in range(trail_number):
            tmp_voice = []
            voice_path = f'{folder_path}/AAD_{dataset_name}/clean/Trail{int(k_tra + 1)}.wav'
            tmp, my_voice = wavfile.read(voice_path)
            my_voice = np.array(my_voice)
            for k_voice_track in range(2):
                voice_track = my_voice[:, k_voice_track]
                voice_track = voice_track.tolist()
                while voice_track[-1] == 0:
                    voice_track.pop()
                tmp_voice.append(voice_track)
            voice.append(tmp_voice)

        # 加载脑电数据
        data_path = f'{folder_path}/AAD_{dataset_name}/S{sub_id}/'
        files = os.listdir(data_path)
        files = sorted(files)  # 按顺序，避免label不同
        for file in files:
            # 输入格式化
            data_mat = scipy.io.loadmat(data_path + file)
            for k_tra in range(data_mat['Markers'].shape[1] // 3):
                k_sta = data_mat['Markers'][0, 3 * k_tra + 2][3][0][0]
                # 避免Trail中断
                if len(data_mat['Markers'][0]) > 3 * k_tra + 3:
                    k_end = data_mat['Markers'][0, 3 * k_tra + 3][3][0][0]
                else:
                    k_end = len(data_mat[channel_names_scut[0]]) - 1

                tmp_eeg = np.zeros((k_end - k_sta, 64))
                for k_cha in range(len(channel_names_scut)):
                    tmp_eeg[:, k_cha] = data_mat[channel_names_scut[k_cha]][k_sta:k_end, 0]
                eeg.append(tmp_eeg)

        label = copy.deepcopy(scut_label)
        voice, label = scut_order(voice, label, sub_id)

        # 处理异常的Trail，比如不完全的Trail等
        eeg, voice, label = scut_remove(eeg, voice, label, sub_id)
    else:
        raise ValueError('数据库名称错误：“dataset_name”不属于已知数据库（KUL、DTU、SCUT）')

    fs_eeg = fs_dict[dataset_name]
    fs_voice = 44100

    # 数据裁剪
    for k_tra in range(len(eeg)):
        data_len = []
        data_len.append(int(eeg[k_tra].shape[0] / fs_eeg))
        data_len.append(int(len(voice[k_tra][0]) / fs_voice))
        data_len.append(int(len(voice[k_tra][1]) / fs_voice))
        data_len.append(55)

        # 计算最短的数据时长（秒）
        min_len = min(data_len)

        # 计算个样本的帧数
        eeg_len = min_len * fs_eeg
        voice_len = min_len * fs_voice

        eeg[k_tra] = eeg[k_tra][0:eeg_len, :]
        voice[k_tra][0] = voice[k_tra][0][0:voice_len]
        voice[k_tra][1] = voice[k_tra][1][0:voice_len]

    # # 去除工频电干扰
    # data = np.transpose(data, [0, 2, 1])
    # data = mne.filter.notch_filter(data, Fs=fs_dict[dataset_name], freqs=50)  # 陷波50Hz
    # data = np.transpose(data, [0, 2, 1])

    return eeg, voice, label


def filter_voice(voice, l_freq, h_freq, fs, is_hilbert=True, is_p_law=True, internal_fs=8000, gl=150, gh=4000, space=1.5):
    """
    语音数据预处理。包含希尔伯特变换、p-law变换和带通滤波（含降采样）
    :param voice: 原始的语音数据，列表形式，每个Trail为Time*2
    :param l_freq: 带通滤波的下限
    :param h_freq: 带通滤波的上限
    :param fs: 语音的输出频率
    :param is_hilbert: 是否进行希尔伯特变换
    :param is_p_law: 是否进行p-law变换
    """
    from eutils.audio import audspacebw, gammatonefir
    fs_voice = 44100

    for k_tra in range(len(voice)):
        my_voice = voice[k_tra]
        my_voice = np.array(my_voice)

        # 降采样
        samples = int(my_voice.shape[1] / fs_voice * internal_fs)
        tmp_voice = [signal.resample(my_voice[0], samples), signal.resample(my_voice[1], samples)]
        my_voice = np.array(tmp_voice)

        # gammatone filter bank
        freqs = audspacebw(gl, gh, space)
        g = gammatonefir(freqs, fs=internal_fs, betamul=space)
        total_voice = []
        for sub in range(len(g[0])):
            sub_voice = signal.filtfilt(g[0][sub].squeeze(), [1], my_voice)

            # 希尔伯特变换
            if is_hilbert:
                sub_voice = hilbert(sub_voice)

            # p-law
            if is_p_law:
                sub_voice = abs(sub_voice)
                sub_voice = np.power(sub_voice, 0.6)

            # 降采样
            samples = int(sub_voice.shape[1] / internal_fs * fs)
            tmp_voice = [signal.resample(sub_voice[0], samples), signal.resample(sub_voice[1], samples)]
            sub_voice = np.array(tmp_voice)

            # 滤波
            wn = [l_freq / fs * 2, h_freq / fs * 2]
            # noinspection PyTupleAssignmentBalance
            b, a = butter(N=8, Wn=wn, btype='bandpass')  # 配置滤波器 8 表示滤波器的阶数
            sub_voice = signal.filtfilt(b, a, sub_voice)  # data为要过滤的信号
            total_voice.append(sub_voice)

        # 合起来
        length = len(total_voice[0][0])
        my_voice = [np.zeros(length), np.zeros(length)]
        for i in range(len(total_voice)):
            my_voice[0] += total_voice[i][0]
            my_voice[1] += total_voice[i][1]

        # 标准化
        my_voice[0] = (my_voice[0] - np.average(my_voice[0])) / np.std(my_voice[0])
        my_voice[1] = (my_voice[1] - np.average(my_voice[1])) / np.std(my_voice[1])
        voice[k_tra] = my_voice

    return voice


def ica_eeg(eeg, dataset_name, label_type):
    """
    对数据进行ICA处理，去伪迹，同时去除50Hz工频干扰
    Args:
        eeg: 原始输入数据
        dataset_name: 数据库的名称
    Returns:
        data: 处理后的数据

    """
    from .preprocess import data_loader as eeg_data_loader

    # ica_dict = {'KUL': [0, 2], 'DTU': ['eog1', 'eog2', 'eog3', 'eog4', 'eog5', 'eog6'], 'SCUT': [0, 1]}
    ica_dict = {'KUL': [0, 2], 'DTU': [0, 1, 7], 'SCUT': [0, 1]}  # 手动选择的ICA成分

    # 准备电极信息
    info = set_info(dataset_name)

    # 加载模板数据（S1-Trail1）
    eeg_tmp, label_tmp = eeg_data_loader(dataset_name, '1', label_type)
    eeg_tmp = eeg_tmp[0]
    eeg_tmp = np.transpose(eeg_tmp, (1, 0))

    # 计算ica通道
    raw_tmp = mne.io.RawArray(eeg_tmp[:, :], info)
    raw_tmp = raw_tmp.filter(l_freq=1, h_freq=None)
    raw_tmp.set_montage(montage_dict[dataset_name])
    ica_tmp = ICA(n_components=20, max_iter='auto', random_state=97)
    ica_tmp.fit(raw_tmp)

    # 去眼电
    is_verbose = True

    for k_tra in range(len(eeg)):
        print(f'data ica, trail: {k_tra}')

        my_eeg = eeg[k_tra]
        my_eeg = np.transpose(my_eeg, [1, 0])

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(my_eeg[:, :], info, verbose=is_verbose)

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

        # 基于ICA去眼电
        ica.exclude = list(set(ica_exclude))
        ica.apply(raw, verbose=is_verbose)
        print(ica.exclude)
        del ica

        # 储存数据
        my_eeg = raw.get_data()
        my_eeg = np.transpose(my_eeg, [1, 0])
        eeg[k_tra] = my_eeg

        # 关闭可视化过程
        is_verbose = False

    return eeg


def filter_eeg(eeg, dataset_name, l_freq, h_freq):
    """
    对数据进行滤波处理，并降低采样率到128Hz（标准化的采样率）
    Args:
        eeg: 去伪迹后的数据
        dataset_name: 数据库名称
        l_freq:带通滤波的低频范围
        h_freq:带通滤波的高频范围

    Returns:
        data: 滤波后的数据

    """

    # # 建立空矩阵储存数据
    # points_num = int(eeg.shape[1] / fs_dict[dataset_name] * 128)
    # # data_resample = np.empty((0, 64, points_num))

    # 滤波
    is_verbose = True
    info = set_info(dataset_name)
    for k_tra in range(len(eeg)):
        print(f'data filter, trail: {k_tra}')

        my_eeg = eeg[k_tra]
        my_eeg = np.transpose(my_eeg, [1, 0])

        # 将原始数据转化为raw格式文件
        raw = mne.io.RawArray(my_eeg, info, verbose=is_verbose)

        # 重参考、滤波、降采样
        # TODO: 添加多频带处理机制
        ref_channels = 'average'
        if dataset_name == "DTU":
            ref_channels = ['ecg1', 'ecg2']
        raw = raw.set_eeg_reference(ref_channels=ref_channels, verbose=is_verbose)
        raw = raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=is_verbose)
        raw = raw.resample(128)


        # 储存数据 标准化
        my_eeg = raw.get_data()[0:64, :]
        my_eeg = (my_eeg - np.average(my_eeg, axis=-1)[..., None]) / np.std(my_eeg, axis=-1)[..., None]
        my_eeg = np.transpose(my_eeg, [1, 0])
        eeg[k_tra] = my_eeg

        # 关闭可视化过程
        is_verbose = False

    return eeg


def signal_split(x, time_len):
    """
    对信号进行分窗
    Args:
        x: list[np.ndarray], len(x) is trails_num, each element shape as [time, channel]
        time_len:

    Returns: 分窗后的数据 shape as [sample, time, channel]

    """
    sample_len = int(128 * time_len)

    # cutoff
    trails_num = len(x)
    for i_tra in range(trails_num):
        points_num, channels_num = x[i_tra].shape
        samples_num = points_num // sample_len
        x[i_tra] = x[i_tra][:samples_num * sample_len, :]
        x[i_tra] = np.reshape(x[i_tra], [samples_num, sample_len, channels_num])

    return x


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

    if dataset_name == 'DTU':
        ch_names = ch_names + ['ecg1', 'ecg2']
        ch_types = ch_types + ['ecg', 'ecg']
        if is_add:
            ch_names = ch_names + ['ecg1', 'ecg2', 'eog1', 'eog2', 'eog3', 'eog4', 'eog5', 'eog6']
            ch_types = ch_types + ['ecg', 'ecg', 'eog', 'eog', 'eog', 'eog', 'eog', 'eog']

    info = mne.create_info(ch_names, fs_dict[dataset_name], ch_types)
    info.set_montage(montage_dict[dataset_name])

    return info


if __name__ == '__main__':
    # data_eeg, data_voice, data_label = preprocess('KUL', '16', 1, 32)
    data_eeg, data_voice, data_label = preprocess_audio('SCUT', sub_id='16', l_freq=1, h_freq=50, is_ica=True)