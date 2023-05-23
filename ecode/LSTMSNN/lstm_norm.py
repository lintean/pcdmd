import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses
from tensorflow.python.keras import backend

from S20_tools.device_to_use import GPU_unable
from S20_tools.subFunCSP import learnCSP

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽通知信息

sample_time_length = 1

str_label = 'Jon0902G1'

# 模型控制参数
dataset = 'KUL'
is_se = False
is_sa = True
is_CSP = False
is_beyond_subject = False  # 同被试是True， 跨被试是False

fold_sta = 0
fold_end = 5 if is_beyond_subject else 1
# 数据库参数
if dataset == 'KUL':
    total_trail = 8
    total_subject = 16
    con_list = ['No']
    batch_size = 64
    wav_channel = 1
    eeg_channel_num = 64
elif dataset == 'DTU':
    total_trail = 20
    total_subject = 18
    con_list = ['No', 'Low', 'High']
    batch_dict = {'0.1': 128, '0.2': 64, '0.5': 32, '1': 16, '2': 16, '5': 8, '10': 4}
    batch_size = batch_dict[str(sample_time_length)]
    eeg_channel_num = 64
elif dataset == 'FAU':
    total_trail = 6
    total_subject = 27
    con_list = ['No']
    batch_size = 64
    wav_channel = 10
    eeg_channel_num = 22

csp_channel_num = 4
model_channels = eeg_channel_num if not is_CSP else 2 * csp_channel_num

fs = 128
band_num = 1  # 目前只能处理单频带，CSP没有改成支持多频带的数据

# 模型细节
lr = 1e-3
overlay_percent = (1 - (1 / math.ceil(sample_time_length))) if sample_time_length > 1 else 0
print('overlap: ', str(overlay_percent))
test_percent = 0.2 if sample_time_length < 10 else 0.25
input_length = int(fs * sample_time_length)
input_channels = eeg_channel_num * band_num
window_length = input_length
window_channels = int(eeg_channel_num * band_num)
output_size = 2  # labels are from 0 to 1
# CNN 参数设置
cnn_kernel_num = 5
cnn_block_len = 4
lstm_units = 10
# SA 参数设置
sa_block_num = math.ceil(sample_time_length)
sa_kq = 100
sa_channel_dense_num = 2 * lstm_units
# SE 参数设置
se_do_percent = 0.5
se_block_len = input_channels

if eeg_channel_num == 64 and (dataset == 'KUL' or dataset == 'DTU'):
    channel_index = []
elif eeg_channel_num == 32 and (dataset == 'KUL' or dataset == 'DTU'):
    channel_index = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24, 27, 29, 31, 32, 34, 36, 38, 40, 42, 44, 46, 48,
                     50, 52, 54, 56, 58, 60, 61]
elif eeg_channel_num == 16 and (dataset == 'KUL' or dataset == 'DTU'):
    channel_index = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 29, 31, 32, 34,
                     35, 36, 38, 40, 41, 42, 43, 44, 45, 46, 48, 50, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62]
elif eeg_channel_num == 10 and (dataset == 'KUL' or dataset == 'DTU'):
    channel_index = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30,
                     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54, 55, 56, 57,
                     58, 61, 62, 63]

if eeg_channel_num == 22 and dataset == 'FAU':
    channel_index = []

# folder name
folder_name = str_label + '_' + str(sa_block_num) + '_' + str(se_block_len) + \
              '_' + str(time.strftime('%H%M', time.localtime(time.time()))) + \
              '_' + dataset + '_cha' + str(eeg_channel_num) + '_band' + str(band_num) + \
              '_sa' + str(is_sa)[0] + '_se' + str(is_se)[0] + '_inSub' + str(is_beyond_subject)[0] + \
              '_len' + str(sample_time_length) + 's' + 'B' + str(batch_size)
print(folder_name)
is_debug = False


class My_SA(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(My_SA, self).__init__(**kwargs)
        # 乘法注意力
        self.dense_k = keras.models.Sequential([
            keras.layers.Dropout(0.5),
            keras.layers.Dense(sa_kq, activation='elu'),
        ])
        self.dense_q = keras.models.Sequential([
            keras.layers.Dropout(0.5),
            keras.layers.Dense(sa_kq, activation='elu'),
        ])
        self.dense_v = keras.models.Sequential([
            keras.layers.Dropout(0.5),
            keras.layers.Dense(sa_channel_dense_num, activation='tanh'),
        ])
        self.my_se_softmax = keras.layers.Softmax(1)

    def build(self, input_shape):
        super(My_SA, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x, **kwargs):
        # eeg shape: (Batch, Time, Channel)
        # 乘法注意力
        k = self.dense_k(x)
        q = self.dense_q(x)
        v = self.dense_v(x)
        w = backend.batch_dot(k, tf.transpose(q, [0, 2, 1])) / math.sqrt(input_length)
        w = self.my_se_softmax(w)
        y = backend.batch_dot(w, v) + x

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


def set_model(k_name):
    # 模型搭建
    model = Sequential()

    # 卷积层
    model.add(keras.layers.BatchNormalization(axis=-1, input_shape=(input_length, input_channels)))

    model.add(keras.layers.Dense(2 * lstm_units, activation='tanh', use_bias=False))
    # 添加SA层
    if is_sa:
        model.add(My_SA())
    model.add(keras.layers.Bidirectional(keras.layers.SLSTM(lstm_units, return_sequences=True, dropout=0.5)))
    model.add(keras.layers.Bidirectional(keras.layers.SLSTM(lstm_units, return_sequences=True, dropout=0.5)))

    # 添加分类器
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(output_size, activation='sigmoid'))

    # 设置优化器参数
    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    # 设置回调函数
    log_dir = "./S20_logs/sub" + str(k_name) + '/' + folder_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=25, verbose=1,
                                                       mode='max', min_delta=1e-6, cooldown=0, min_lr=1e-6)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=50, verbose=1,
                                                           mode='min', restore_best_weights=False)
    # 可视化输出
    if is_tensorboard:
        callbacks = [tensorboard_callback, lr_callback, early_stop_callback]
        print('callbacks: tensorboard, lr, early stop')
    else:
        callbacks = [early_stop_callback, lr_callback]
        print('callbacks: lr, early stop')
    return callbacks, model


def main(k_name, epochs, k_fold):
    print('k_name:', k_name)
    print('epochs:', epochs)
    print('k_fold: ', k_fold)

    x_test, x_train, x_val, y_test, y_train, y_val = data_set(k_name, k_fold)

    # 模型设置
    callbacks, model = set_model(k_name)

    # model(x_train)
    model.summary()
    model.fit(x_train, y_train, epochs=epochs, verbose=2, batch_size=batch_size, callbacks=callbacks,
              validation_freq=1, validation_data=(x_val, y_val), shuffle=True)

    print('end')
    loss, acc = model.evaluate(x_test, y_test)
    print(acc)


def csp_dot(w_csp, x):
    # CSP处理过程
    tmp_x = np.empty([x.shape[0], x.shape[1], csp_channel_num * 2, 0])
    # TODO: 多频带处理
    # 准备当前csp矩阵
    tmp_w_csp = w_csp
    tmp_w_csp = np.concatenate([tmp_w_csp[:, 0:csp_channel_num], tmp_w_csp[:, -csp_channel_num:]], axis=1)
    csp_x = np.matmul(x[:, :, :], tmp_w_csp)
    tmp_x = np.concatenate([tmp_x, tf.expand_dims(csp_x, axis=-1)], axis=-1)

    return tmp_x


def data_set(k_name, k_fold):
    global dataset, con_list, total_subject, total_trail

    # 数据加载
    if is_beyond_subject:
        x_test, x_train, x_val, y_test, y_train, y_val = load_data(k_name, input_channels, k_fold)
    else:
        # 训练集、验证集、测试集分配
        train_index = list(range(1, total_subject + 1))
        test_index = k_name
        del train_index[k_name - 1]
        val_index = random.sample(train_index, 5)
        for tmp_k in val_index:
            train_index.remove(tmp_k)

        # 测试集
        x_test, y_test = sub_csp(test_index, True, k_fold)

        # 验证集
        x_val = np.empty([0, sample_time_length * fs, eeg_channel_num])
        y_val = np.empty([0])
        for sub in val_index:
            x_tmp, y_tmp = sub_csp(sub, False, k_fold)
            x_val = np.concatenate([x_val, x_tmp], axis=0)
            y_val = np.concatenate([y_val, y_tmp], axis=0)

        # 训练集
        x_train = np.empty([0, sample_time_length * fs, eeg_channel_num])
        y_train = np.empty([0])
        for sub in train_index:
            print('sub: ', str(sub))
            x_tmp, y_tmp = sub_csp(sub, False, k_fold)
            x_train = np.concatenate([x_train, x_tmp], axis=0)
            y_train = np.concatenate([y_train, y_tmp], axis=0)

        # # 添加其他数据库的人(结果证明有效)
        # if dataset == 'DTU':
        #     dataset = 'KUL'
        # else:
        #     dataset = 'DTU'
        #
        # if dataset == 'KUL':
        #     total_trail = 8
        #     total_subject = 16
        #     con_list = ['No']
        # elif dataset == 'DTU':
        #     total_trail = 20
        #     total_subject = 18
        #     con_list = ['No', 'Low', 'High']
        #
        # for sub in range(total_subject):
        #     print('sub2: ', str(sub))
        #     x_tmp, y_tmp = sub_csp(sub + 1, False, k_fold)
        #     x_train = np.concatenate([x_train, x_tmp], axis=0)
        #     y_train = np.concatenate([y_train, y_tmp], axis=0)

    # TODO： CSP 数据处理方法
    if is_CSP:
        w_csp = learnCSP(x_train, y_train)
        # csp 处理数据
        x_train = csp_dot(w_csp, x_train)
        x_test = csp_dot(w_csp, x_test)
        x_val = csp_dot(w_csp, x_val)

    # 输入数据预处理
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    x_val, y_val = preprocess(x_val, y_val)
    return x_test, x_train, x_val, y_test, y_train, y_val


def window_split(data, label, x, y):
    data_len = data.shape[0]
    data_channel = data.shape[1]
    if overlay_percent == 0:
        index_s = list(range(0, data_len - window_length, window_length))
        index_e = list(range(window_length, data_len, window_length))
    else:
        index_s = list(range(0, data_len - window_length, int(window_length * (1 - overlay_percent))))
        index_e = list(range(window_length, data_len, int(window_length * (1 - overlay_percent))))

    data = np.array(data)
    window_data = np.empty((0, window_length, data_channel))
    for k in range(len(index_s)):
        temp_data = data[index_s[k]:index_e[k]]
        window_data = np.concatenate((window_data, temp_data[np.newaxis, :]), axis=0)

    window_label = np.ones(window_data.shape[0]) * label - 1

    x = np.concatenate([x, window_data], axis=0)
    y = np.concatenate([y, window_label], axis=0)
    return x, y


def data_split(data, k_fold):
    data_len = int(data.shape[0])
    test_len = int(data.shape[0] * test_percent) - 1
    test_index_s = int(k_fold * (data_len / 5)) + 1
    test_index_e = test_index_s + test_len
    if sample_time_length > 5 and dataset == 'DTU':
        print('Note: data length is not enough, add some!')
        add_len = fs * 5
        if k_fold < 4:
            test_index_e = test_index_s + test_len + add_len
        else:
            test_index_s = int(k_fold * (data_len / 5)) + 1 - add_len

    test_data = data[test_index_s:test_index_e]
    train_data = pd.concat([data[0:test_index_s], data[test_index_e:]], axis=0)

    return test_data, train_data


def load_data(k_name, data_channel, k_fold):
    k_name = int(k_name)
    # 数据存储
    x_train = np.empty((0, window_length, data_channel))
    x_test = np.empty((0, window_length, data_channel))
    x_val = np.empty((0, window_length, data_channel))
    y_train = np.empty(0)
    y_test = np.empty(0)
    y_val = np.empty(0)

    # path set
    csv_path = './S10_data/' + dataset + '_B' + str(band_num) + '_1to32' + '/'
    for tmp_con in con_list:
        # 注意力标签
        if dataset == 'FAU':
            label_name = csv_path + 'label/S' + str(k_name).zfill(2) + tmp_con + '.csv'
        else:
            label_name = csv_path + 'label/S' + str(k_name) + tmp_con + '.csv'
        label_sub = pd.read_csv(label_name, header=None)

        for k_trail in range(total_trail):
            file_name = csv_path + tmp_con + '/S' + str(k_name).zfill(2) + 'Tra' + str(k_trail + 1) + '.csv'
            data = pd.read_csv(file_name)

            data = data.drop(data.columns[list(range(wav_channel * 2))], axis=1)

            # 减少通道数量
            data = data.drop(data.columns[channel_index], axis=1)

            # 将整个Trail的数据分为三份
            test_data, train_data = data_split(data, k_fold)
            val_data, train_data = data_split(train_data, abs(4 - k_fold))
            del data

            label = label_sub[0][k_trail + 1]

            # 合并不同Trail的训练集、验证集、测试集
            x_train, y_train = window_split(train_data, label, x_train, y_train)
            x_test, y_test = window_split(test_data, label, x_test, y_test)
            x_val, y_val = window_split(val_data, label, x_val, y_val)

    return x_test, x_train, x_val, y_test, y_train, y_val


def sub_csp(test_index, test_type, k_fold):

    tmp_x_test, tmp_x_train, tmp_x_val, tmp_y_test, tmp_y_train, tmp_y_val = load_data(test_index, input_channels,
                                                                                       k_fold)
    if test_type:
        x_test = np.concatenate([tmp_x_train, tmp_x_test, tmp_x_val], axis=0)
        y_test = np.concatenate([tmp_y_train, tmp_y_test, tmp_y_val], axis=0)
    else:
        x_test = tmp_x_train
        y_test = tmp_y_train
    # x_test = csp_dot(w_csp, x_test)
    return x_test, y_test


def preprocess(x, y):
    x, y = shuffle(x, y, random_state=random.randint(0, 1000))
    x = tf.reshape(x, [x.shape[0], window_length, model_channels])
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=2)
    return x, y


def gpu_set():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    for k in range(len(GPU_unable)):
        memory_gpu[GPU_unable[k]] = 0
    gpu_random = str(memory_gpu.index(max(memory_gpu)))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_random
    print('gpu: ', gpu_random)
    # 跨被试相关程序，未完成。
    # if is_beyond_subject:
    #     gpu_random = str(memory_gpu.index(max(memory_gpu)))
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_random
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    # 设置gpu内存自增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        is_tensorboard: bool = False
        is_debug = False

        # 设置使用的GPU
        gpu_set()
        main(k_name=int(sys.argv[1]), epochs=int(sys.argv[2]), k_fold=int(sys.argv[3]))
    else:
        is_tensorboard: bool = True
        # is_debug = True
        if is_debug:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置使用GPU，CPU为-1

        # # 设置使用的GPU
        # gpu_set()

        main(1, 100, k_fold=0)
