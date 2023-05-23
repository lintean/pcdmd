from scipy.signal import hilbert
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torch.autograd import Function
from torch.autograd import Variable
import math
import random
import sys
import os
from parameters import *


def add_delay(data):
    sound = data.iloc[:, 0]
    sound_not_target = data.iloc[:, 1]
    EEG = data.iloc[:, 2:]

    if delay >= 0:
        sound = sound.iloc[:sound.shape[0] - delay]
        sound_not_target = sound_not_target.iloc[:
                                                 sound_not_target.shape[0] - delay]
        EEG = EEG.iloc[delay:, :]
    else:
        sound = sound.iloc[-delay:]
        sound_not_target = sound_not_target.iloc[-delay:]
        EEG = EEG.iloc[:EEG.shape[0] + delay, :]

    sound = sound.reset_index(drop=True)
    sound_not_target = sound_not_target.reset_index(drop=True)
    EEG = EEG.reset_index(drop=True)

    data_pf = pd.concat([sound, sound_not_target, EEG],
                        axis=1, ignore_index=True)
    return data_pf

# filter band index
def read(name, fb_index):
    first = True
    data_pf = pd.DataFrame()
    data = pd.DataFrame()

    for l in range(len(ConType)):
        sex = pd.read_csv("./csv/" + name + ConType[l] + ".csv")
        for k in range(20):
            # 读取数据
            train_data = pd.read_csv(data_document_path + "/" +
                                     ConType[l] + "/" + name + "Tra" + str(k + 1) + "_" + str(fb_index) + ".csv", header=None)
            EEG_data = train_data.iloc[:, 2:]
            Sound_data = train_data.iloc[:, 0]
            Sound_data_not_target = train_data.iloc[:, 1]

            # 调整左右位置，添加辅助信息
            if isDS and sex.iloc[k, isFM] == 2:
                temp = Sound_data
                Sound_data = Sound_data_not_target
                Sound_data_not_target = temp

            # 合并
            EEG_data = pd.DataFrame(EEG_data)
            Sound_data = pd.DataFrame(Sound_data)
            Sound_data_not_target = pd.DataFrame(Sound_data_not_target)
            data_pf = pd.concat(
                [Sound_data, Sound_data_not_target, EEG_data], axis=1, ignore_index=True)

            # 加入时延
            data_pf = add_delay(data_pf)

            if first:
                data = data_pf
                first = False
            else:
                data = pd.concat([data, data_pf], axis=0, ignore_index=True)

    return data


def timeSplit(data, name):
    def update_CNN_DS_S(data_pf, data_direction, temp_data, temp_direction):
        # 如果是CNN：D+S或CNN：FM+S模型
        if isDS:
            data_pf = pd.concat([data_pf, temp_data],
                                axis=0, ignore_index=True)
            data_direction.append(temp_direction)
        # 否则是CNN：S模型
        else:
            temp = np.array(temp_data)
            data_pf = pd.concat([data_pf, pd.DataFrame(
                temp.copy())], axis=0, ignore_index=True)
            temp[:, [0, channle_number - 1]] = temp[:, [channle_number - 1, 0]]
            data_pf = pd.concat([data_pf, pd.DataFrame(
                temp.copy())], axis=0, ignore_index=True)
            data_direction.append(1)
            data_direction.append(2)
        return data_pf, data_direction

    # 参数初始化
    global cell_number
    global test_percent
    cell_number = cell_number - abs(delay)
    window_lap = window_length * (1 - overlap)
    overlap_distance = math.floor(1 / (1 - overlap)) - 1
    selection_trails = 0
    if isBeyoudTrail:
        selection_trails = random.sample(
            range(trail_number), math.ceil(trail_number * test_percent))

    # 找不到其他空矩阵创建方法，先用着
    data_pf = pd.DataFrame(data.iloc[:window_length, :])
    test_pf = pd.DataFrame(data.iloc[:window_length, :])
    data_direction = []
    test_direction = []

    # 对于每个ConType进行划分
    for l in range(len(ConType)):
        sex = pd.read_csv("./csv/" + name + ConType[l] + ".csv")

        # 对于ConType里的每个trail进行划分
        for k in range(trail_number):
            # 每个trail里的窗口个数
            window_number = math.floor(
                (cell_number - window_length) / window_lap) + 1
            # 随机抽取的测试窗口长度
            if isBeyoudTrail:
                test_percent = 1 if k in selection_trails else 0
            test_percent = 0 if isALLTrain else test_percent
            test_window_length = math.floor(
                (cell_number * test_percent - window_length) / window_lap)
            test_window_length = test_window_length if test_percent == 0 else max(
                0, test_window_length)
            test_window_length = test_window_length + 1
            # 随机抽取的测试窗口左右边界
            test_window_left = random.randint(
                0, window_number - test_window_length)
            test_window_right = test_window_left + test_window_length - 1

            # 对于ConType里的trail里的每个窗口进行划分
            for i in range(window_number):
                left = math.floor(k * cell_number + i * window_lap)
                right = math.floor(left + window_length)
                # 如果不是要抽取的测试窗口，即为训练集里的窗口
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    temp_data = data.iloc[left:right, :]
                    temp_direction = sex.iloc[k, isFM]
                    data_pf, data_direction = update_CNN_DS_S(
                        data_pf, data_direction, temp_data, temp_direction)
                elif i >= test_window_left and i <= test_window_right:
                    temp_data = data.iloc[left:right, :]
                    temp_direction = sex.iloc[k, isFM]
                    test_pf, test_direction = update_CNN_DS_S(
                        test_pf, test_direction, temp_data, temp_direction)

    # 去除初始化的数据
    data_pf = data_pf.iloc[window_length:, :]
    test_pf = test_pf.iloc[window_length:, :]

    # 重新组织结构
    data_pf = np.array(data_pf).reshape(-1, window_length, channle_number)
    test_pf = np.array(test_pf).reshape(-1, window_length, channle_number)

    data = []
    for i in range(data_pf.shape[0]):
        d = dict()
        d["data"] = data_pf[i]
        d["direction"] = data_direction[i]
        data.append(d)

    test = []
    for i in range(test_pf.shape[0]):
        d = dict()
        d["data"] = test_pf[i]
        d["direction"] = test_direction[i]
        test.append(d)

    # return np.array(data), np.array(test)
    # 这里先让它输出list方便append
    return data, test


def change(train_data):
    train_sound = train_data.iloc[:, 0]
    train_sound_not_target = train_data.iloc[:, 1]
    train_EEG = train_data.iloc[:, 2:]
    data_pf = pd.concat(
        [train_sound, train_EEG, train_sound_not_target], axis=1, ignore_index=True)
    return data_pf


def main(name = "S18"):
    print("start!")

    # 提取不同频带数据并汇总得到训练集和测试集
    train = []
    test = []
    for i in range(5):
        # 读取数据
        data_pf = read(name, i+1)
        # 调整数据结构
        data_pf = change(data_pf)
        # 划分时间窗口并生成训练集和测试集
        newtrain, newtest = timeSplit(data_pf, name)
        train += newtrain
        test += newtest
    # random.shuffle(train)
    # random.shuffle(test)
    train = np.array(train)
    test = np.array(test)

    # 合并和保存结果
    data_pf = [train, test]
    data_pf = np.array(data_pf)
    np.save("./data_new/CNN1_" + name, data_pf)

    print("finish!")


if __name__ == "__main__":
    if (len(sys.argv) == 2):
        main(sys.argv[1])
    else:
        main()
