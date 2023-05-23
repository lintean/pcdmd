#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   db_preprocess.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/8 10:51   lintean      1.0         None
'''

from db.SCUT.preproc_update import preprocess as scut_prep


def preprocess(dataset_name, *args, **kwargs):
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

    datasets = ["KUL", "DTU", "SCUT"]
    preps = [None, None, scut_prep]
    for i in range(len(datasets)):
        if dataset_name == datasets[i]:
            return preps[i](dataset_name, *args, **kwargs)
