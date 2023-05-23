import keras
import numpy as np
from math import ceil
import eutils.util as util
import math
import tensorflow as tf


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, metadata, *args, **kwargs):
        """
           self.data_dir:数据在电脑中存放的路径。
        """
        self.data = data
        self.metadata = metadata
        self.window_number = data.shape[0]
        self.pos_emb = self.positional_embedding(metadata.batch_size, metadata.eeg_channel * metadata.netSize)
        self.list_IDs = [i for i in range(ceil((self.window_number - metadata.netSize + 1) / metadata.batch_size))]
        self.on_epoch_end()

    def __len__(self):
        """
           返回生成器的长度，也就是总共分批生成数据的次数。
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
           该函数返回每次我们需要的经过处理的数据。
        """
        eegs = []
        sounds = []
        start = self.get_data_index(self.list_IDs[index])
        for i in range(self.metadata.batch_size):
            if start + i + self.metadata.netSize > self.data.shape[0]:
                break
            data_temp = self.data[start + i:start + i + self.metadata.netSize, :]
            sound, EEG, sound_not_target = util.get_split_data(data_temp, self.metadata)
            eegs.append(EEG.flatten())
            sounds.append(sound[0])
        eegs = np.stack(eegs, axis=0)
        sounds = np.stack(sounds, axis=0)

        # if eegs.shape != (16, 1728):
        #     print(eegs)
        #     input()

        # if eegs.shape[0] != self.pos_emb.shape[0]:
        #     eegs = eegs + self.positional_embedding(eegs.shape[0], eegs.shape[1])
        # else:
        #     eegs = eegs + self.pos_emb

        return eegs, sounds

    def on_epoch_end(self):
        """
           该函数将在训练时每一个epoch结束的时候自动执行，在这里是随机打乱索引次序以方便下一batch运行。
        """
        np.random.shuffle(self.list_IDs)

    def get_data_index(self, index):
        return self.metadata.batch_size * index

    # 位置编码信息
    def positional_embedding(self, maxlen, model_size):
        PE = np.zeros((maxlen, model_size))
        for i in range(maxlen):
            for j in range(model_size):
                if j % 2 == 0:
                    PE[i, j] = np.sin(i / 10000 ** (j / model_size))
                else:
                    PE[i, j] = np.cos(i / 10000 ** ((j - 1) / model_size))
        return PE

class TestGenerator(keras.utils.Sequence):

    def __init__(self, data, metadata, *args, **kwargs):
        self.data = data
        self.metadata = metadata
        self.window_number = data.shape[0]
        self.list_IDs = [i for i in range(ceil((self.window_number - metadata.netSize + 1) / 1))]

    def __len__(self):
        """
           返回生成器的长度，也就是总共分批生成数据的次数。
        """
        return 1

    def __getitem__(self, index):
        """
           该函数返回每次我们需要的经过处理的数据。
           目前只考虑batch_size为1的情况
        """
        eegs = []
        start = self.get_data_index(self.list_IDs[index])
        for i in range(len(self.list_IDs)):
            if start + i + self.metadata.netSize > self.data.shape[0]:
                break
            data_temp = self.data[start + i:start + i + self.metadata.netSize, :]
            sound, EEG, sound_not_target = util.get_split_data(data_temp, self.metadata)
            eegs.append(EEG.flatten())
        eegs = np.stack(eegs, axis=0)

        # eegs = [eegs for i in range(self.metadata.numSampleContext)]
        return eegs

    def get_data_index(self, index):
        return index
