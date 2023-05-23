import numpy as np


def data_split(data, label, sample_len):
    """
    数据结构的预处理
    1. 改变数据划分方式；
    2. 对标签进行独热编码；
    :param data: trails*time*channels
    :param label: trails
    :param sample_len: the points number of each sample, for example 128Hz with 1s
    :return:
        data: N * length * channels
        label: N * 2(类别)
    """
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
