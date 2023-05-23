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

def add_delay(data, delay):
    sound = data.iloc[:, 0]
    sound_not_target = data.iloc[:, 1]
    EEG = data.iloc[:, 2:]
    
    if delay >= 0:
        sound = sound.iloc[:sound.shape[0] - delay]
        sound_not_target = sound_not_target.iloc[:sound_not_target.shape[0] - delay]
        EEG = EEG.iloc[delay:, :]
    else:
        sound = sound.iloc[-delay:]
        sound_not_target = sound_not_target.iloc[-delay:]
        EEG = EEG.iloc[:EEG.shape[0] + delay, :]
    
    sound = sound.reset_index(drop=True)
    sound_not_target = sound_not_target.reset_index(drop=True)
    EEG = EEG.reset_index(drop=True)

    data_pf = pd.concat([sound, sound_not_target, EEG], axis=1, ignore_index=True)
    return data_pf

def read(name, ConType, delay, isDS, isFM):
    first = True
    data_pf = pd.DataFrame()
    data = pd.DataFrame()
    sex = pd.read_csv("./csv/" + name + ConType + ".csv")
    
    for k in range(20):
        # 读取数据
        train_data = pd.read_csv("../dataset_csp/" + ConType + "/" + name + "Tra" + str(k + 1) + ".csv", header=None)
        
        EEG_data = train_data.iloc[:, 2:]
        Sound_data = train_data.iloc[:, 0]
        Sound_data_not_target = train_data.iloc[:, 1]
        
        # 调整左右位置(cnn1)
        if isDS == "DS" and sex.iloc[k, isFM] == 2:
            temp = Sound_data
            Sound_data = Sound_data_not_target
            Sound_data_not_target = temp
        
        # 合并
        EEG_data = pd.DataFrame(EEG_data)
        Sound_data = pd.DataFrame(Sound_data)
        Sound_data_not_target = pd.DataFrame(Sound_data_not_target)
        data_pf = pd.concat([Sound_data, Sound_data_not_target, EEG_data], axis=1, ignore_index=True)
        
        # 加入时延
        data_pf = add_delay(data_pf, delay)
        
        if first:
            data = data_pf
            first = False
        else:
            data = pd.concat([data, data_pf], axis=0, ignore_index=True)
            
    return data

def timeSplit(data,name, ConType, overlap, window_length, delay, isDS, isFM):
    def update_CNN_DS_S(data_pf, data_direction, temp_data, temp_direction, isDS):
        if isDS == "DS":
            data_pf = pd.concat([data_pf, temp_data], axis=0, ignore_index=True)
            data_direction.append(temp_direction)
        else:
            temp = np.array(temp_data)
            data_pf = pd.concat([data_pf, pd.DataFrame(temp.copy())], axis=0, ignore_index=True)
            temp[:, [0,65]] = temp[:, [65,0]]
            data_pf = pd.concat([data_pf, pd.DataFrame(temp.copy())], axis=0, ignore_index=True)
            data_direction.append(1) 
            data_direction.append(2)      
        return data_pf, data_direction
    
    sex = pd.read_csv("./csv/" + name + ConType + ".csv")
    overlap = float(overlap)
    window_length = int(window_length)
    channle_number = 66
    trail_number = 20
    cell_number = 3500
    test_percent = 0.1
    delay = abs(delay)
    
    cell_number = cell_number - delay
    window_lap = window_length * (1 - overlap)
    overlap_distance = math.floor(1 / (1 - overlap)) - 1
    
    # 找不到其他空矩阵创建方法，先用着
    data_pf = pd.DataFrame(data.iloc[:window_length, :])
    test_pf = pd.DataFrame(data.iloc[:window_length, :])
    data_direction = []
    test_direction = []
    
    for k in range(trail_number):
        # 每个trail里的窗口个数
        window_number = math.floor((cell_number - window_length) / window_lap) + 1
        #随机抽取的测试窗口长度
        test_window_length = max(0, math.floor((cell_number * test_percent - window_length) / window_lap)) + 1
        # 随机抽取的测试窗口(这里25秒特化)
#         test_window_left = random.randint(0, window_number - test_window_length)
        test_window_left = random.randint(0, 1)
        test_window_left = test_window_left * 2
        test_window_right = test_window_left + test_window_length - 1
        
        for i in range(window_number):
            left = math.floor(k * cell_number + i * window_lap)
            right = math.floor(left + window_length)
            # 如果不是要抽取的测试窗口
            if test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                temp_data = data.iloc[left:right, :]
                temp_direction = sex.iloc[k, isFM]
                data_pf, data_direction = update_CNN_DS_S(data_pf, data_direction, temp_data, temp_direction, isDS)
            elif i >= test_window_left and i <= test_window_right:           
                temp_data = data.iloc[left:right, :]
                temp_direction = sex.iloc[k, isFM]
                test_pf, test_direction = update_CNN_DS_S(test_pf, test_direction, temp_data, temp_direction, isDS)
    
    # 去除初始化的数据
    data_pf = data_pf.iloc[window_length:, :]
    test_pf = test_pf.iloc[window_length:, :]
    
    #重新组织结构
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
    
    return np.array(data), np.array(test)

def change(train_data):
    train_sound = train_data.iloc[:, 0]
    train_sound_not_target = train_data.iloc[:, 1]
    train_EEG = train_data.iloc[:, 2:]
    data_pf = pd.concat([train_sound, train_EEG, train_sound_not_target], axis=1, ignore_index=True)
    return data_pf


def main(name = "all", ConType = "No", overlap = "0.5", window_length = "1740", delay = "0", 
    isDS = "DS", isFM = "0", max_epoch = "1200", batch_size = "1", shutdown_float = "0.32", need_split = "True", Continue = "False"):

    # isFM为0是男女信息，为1是方向信息
    # shutdown_float已弃用，目前用的early stop
    once = True
    if name == "all":
        once = False
    delay = int(delay)
    need_split = need_split == "True"
    isFM = int(isFM)

    label =  isDS + "_" + name + "_" + ConType + "_" + overlap + "_" + window_length + "_" + str(delay)\
        + "_" + batch_size + "_" + max_epoch
    if Continue == "True": 
        os.system("mkdir " + label)
        
    for i in range(18):            
        if not once:
            name = "S" + str(i + 1)

        if need_split:
            # 读取数据
            data_pf = read(name, ConType, delay, isDS, isFM)

            # 调整数据结构
            data_pf = change(data_pf)

            #划分时间窗口并生成训练集和测试集
            train, test = timeSplit(data_pf, name, ConType, overlap, window_length, delay, isDS, isFM)

            # 混入Low数据
            ConType = "Low"
            data_low = read(name, ConType, delay, isDS, isFM)
            data_low = change(data_low)
            train_low, test_low = timeSplit(data_low, name, ConType, overlap, window_length, delay, isDS, isFM)


            # 混入High数据
            ConType = "High"
            data_high = read(name, ConType, delay, isDS, isFM)
            data_high = change(data_high)
            train_high, test_high = timeSplit(data_high, name, ConType, overlap, window_length, delay, isDS, isFM)

            train = np.concatenate((train, train_low, train_high),axis=0)
            test = np.concatenate((test, test_low, test_high),axis=0)
            ConType = "No"
            
            print(train.shape)
            print(test.shape)

            #合并和保存结果
            data_pf = [
                train,
                test
            ]

            data_pf = np.array(data_pf)
            np.save("./data_new/CNN1_" + name, data_pf) 
            

        if Continue == "True": 
            os.system("nohup /home/lipeiwen/anaconda3/bin/python -u CNN.py " + name + " " + max_epoch + " " + batch_size + " " + 
                shutdown_float + " " + overlap + " > " + "./" + label + "/CNN_" + name + "_" + label + ".log 2>&1 &")
        
        print("finish!")

        if once:
            break

if __name__=="__main__": 
    if (len(sys.argv) == 13):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], 
            sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12])
    else:
        main()