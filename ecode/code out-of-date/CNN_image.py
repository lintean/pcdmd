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
from parameters import *
import pickle

def train(train): 
    losses = 0
            
    for turn in range(train['images'].shape[0]):
        optimzer.zero_grad()
        eeg = torch.tensor(train['images'][turn], dtype=torch.float32).to(device)
        audio = torch.tensor(train['audio'][turn], dtype=torch.float32).to(device)
        wavA = audio[:, 0:1]
        wavB = audio[:, 1:2]
        wav = wavA if train['labels'][turn] == 0 else wavB
        wav_alter = wavB if train['labels'][turn] == 0 else wavA
        out = myNet(wav, wav_alter, eeg)
        loss = loss_func(out, torch.tensor(train['labels'][turn][0], dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        loss.backward()
        optimzer.step()
    # scheduler.step()
    scheduler.step(metrics=0.1)
        
    return losses / train['images'].shape[0]

def test(cv):
    losses = 0
            
    for turn in range(cv['images'].shape[0]):
        optimzer.zero_grad()
        eeg = torch.tensor(cv['images'][turn], dtype=torch.float32).to(device)
        audio = torch.tensor(cv['audio'][turn], dtype=torch.float32).to(device)
        wavA = audio[:, 0:1]
        wavB = audio[:, 1:2]
        wav = wavA if cv['labels'][turn] == 0 else wavB
        wav_alter = wavB if cv['labels'][turn] == 0 else wavA
        out = myNet(wav, wav_alter, eeg)
        loss = loss_func(out, torch.tensor(cv['labels'][turn][0], dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        
    return losses / cv['images'].shape[0]

def trainEpoch(data, test_data):
    min_loss = 100
    early_stop_number = 0

    for epoch in range(max_epoch):
        # 打乱非测试数据集并划分训练集和验证集
        train_data = data['train']
        cv_data = data['valid']
        
        index = [i for i in range(train_data['images'].shape[0])]
        random.shuffle(index)
        train_data_i = train_data['images'][index, :, :, :].copy()
        train_data_a = train_data['audio'][index, :, :].copy()
        train_data_l = train_data['labels'][index, :, :].copy()
        train_data = {
            'images' : train_data_i,
            'audio' : train_data_a,
            'labels' : train_data_l
        }
        
        loss_train = train(train_data)
        loss = test(cv_data)
        loss2 = test(test_data)
        
        print(str(epoch) + " " + str(loss_train) + " " + str(loss) + " " + str(loss2), end="")

        if loss > min_loss:
            early_stop_number = early_stop_number + 1
        else:
            early_stop_number = 0
            min_loss = loss

        print(" early_stop_number: ", end="")
        print(early_stop_number, end="")
        print()

        if isEarlyStop and epoch > min_epoch and early_stop_number >= 10:
            break

def testEpoch(test_data):
    total_t_num = 0
    total_f_num = 0
    for num in range(10):
        t_num = 0
        f_num = 0
        for turn in range(test_data['images'].shape[0]):
            optimzer.zero_grad()
            eeg = torch.tensor(test_data['images'][turn], dtype=torch.float32).to(device)
            audio = torch.tensor(test_data['audio'][turn], dtype=torch.float32).to(device)
            wavA = audio[:, 0:1]
            wavB = audio[:, 1:2]
            wav = wavA if test_data['labels'][turn] == 0 else wavB
            wav_alter = wavB if test_data['labels'][turn] == 0 else wavA
            out = myNet(wav, wav_alter, eeg)

            lossL = loss_func(out, torch.tensor([0]).to(device)).cpu().detach().numpy()
            lossR = loss_func(out, torch.tensor([1]).to(device)).cpu().detach().numpy()
            if (lossL<lossR) == (test_data['labels'][turn][0] == 0):
                t_num = t_num + 1
            else:
                f_num = f_num + 1

        print(str(t_num) + " " + str(f_num))
        total_t_num = total_t_num + t_num
        total_f_num = total_f_num + f_num
    print(str(total_t_num / (total_t_num + total_f_num)))


def main(name = "S2"):
    # 参数init
    name_number = int(name[1:])

    # 先读取测试数据
    f = open(data_document_path + "/final_data_subject" + str(name_number) + ".pkl", "rb")
    data = pickle.load(f)
    test_data = data['test']

#     # 读取数据并预训练
#     if need_pretrain:
#         print("pretrain start!")
#         basic_name = "S" + str(name_number % (people_number - 1) + 1)
#         b = np.load("./data_new/CNN1_"+ basic_name + ".npy", allow_pickle=True)
#         for k in range(people_number):
#             filelable = "S" + str(k+1)
#             if (not isALLTrain or filelable != name) and filelable != basic_name:
#                 # 读取数据
#                 a = np.load("./data_new/CNN1_"+ filelable + ".npy", allow_pickle=True)
#                 b[0] = np.hstack((a[0], b[0]))
#                 b[1] = np.hstack((a[1], b[1]))
#         data = b
#         trainEpoch(data, test_data)
#         print()

    # 读取数据并训练
    if need_train:
        # 降低学习率
        if need_pretrain:
            for p in optimzer.param_groups:
                p['lr'] *= 0.1

        print("train start!")
        trainEpoch(data, test_data)
        print()

    # 测试
    print("test start!")
    testEpoch(test_data)

if __name__=="__main__": 
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1])
    else:
        main()