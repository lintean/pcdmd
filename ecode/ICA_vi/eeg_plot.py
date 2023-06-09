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
from eutils.preproc.backup.preprocess import data_loader, set_info, data_ica, data_filter


def makePath(path):
    if not os.path.isdir(path):  # 如果路径不存在
        os.makedirs(path)
    return path


def main(sub_id='1', k_tra=6):
    mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')
    dataset_name = "KUL"
    l_freq = [1]
    h_freq = [32]
    label_type = 'direction'

    data, label = data_loader(dataset_name, sub_id, label_type)
    data = data_ica(data, dataset_name, label_type)
    data = data_filter(data, dataset_name, l_freq, h_freq)

    # 准备电极信息
    info = set_info(dataset_name)
    data = np.transpose(data, (0, 2, 1))

    chan = [
        'FT7', 'FT8', 'P7', 'P9', 'T7',
        'T8', 'TP7', 'TP8', 'P8', 'P10'
    ]
    order = [info.ch_names.index(c) for c in chan]

    raw = mne.io.RawArray(data[k_tra], info, verbose=False)
    fig = raw.plot(
        duration=1, n_channels=10, show=True, show_scrollbars=False,
        color='darkblue',order=order, scalings=5, butterfly=True, show_scalebars=False
    )

    fig.savefig(f"{makePath(f'./eeg_plot')}/2.png", format='png', transparent=True)

    #
    # for index, label in enumerate(labels):
    #     if label == 'brain':
    #         continue
    #     fig = ica.plot_components(picks=[index], title="", size=3.0)
    #     fig.savefig(f"{makePath(f'./ica_plot/{sub_id}')}/{label}-{index}.png", format='png', transparent=True)


if __name__ == "__main__":
    main()
