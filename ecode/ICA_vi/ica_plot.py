#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ica_plot.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/27 21:09   lintean      1.0         None
'''
import mne
import os
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
from eutils.preproc.backup.preprocess import data_loader, set_info, montage_dict


def makePath(path):
    if not os.path.isdir(path):  # 如果路径不存在
        os.makedirs(path)
    return path


def main(sub_id='1'):
    dataset_name = "KUL"
    l_freq = 1
    h_freq = 32
    label_type = 'direction'

    data, label = data_loader(dataset_name, sub_id, label_type)

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
    # for k_tra in range(data.shape[0]):
    k_tra = 0
    print(f'data ica, trail: {k_tra}')

    # 将原始数据转化为raw格式文件
    raw = mne.io.RawArray(data[k_tra], info, verbose=is_verbose)

    # 计算ica数据
    raw = raw.filter(l_freq=1, h_freq=None, verbose=is_verbose)
    ica = ICA(n_components=20, max_iter='auto', random_state=97, verbose=is_verbose)  # 97为随机种子
    ica.fit(raw)

    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels['labels']

    for index, label in enumerate(labels):
        if label == 'brain':
            continue
        fig = ica.plot_components(picks=[index], title="", size=3.0)
        fig.savefig(f"{makePath(f'./ica_plot/{sub_id}')}/{label}-{index}.png", format='png', transparent=True)


if __name__ == "__main__":
    # main()
    for sub in range(1, 17):
        main(f'{sub}')
