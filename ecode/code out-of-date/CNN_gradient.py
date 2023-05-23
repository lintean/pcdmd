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

def vali_split(train):
    window_number = train.shape[0]
    # 随机抽取的验证窗口长度  
    vali_window_length = math.floor(window_number * vali_percent)
    # 随机抽取的验证窗口
    vali_window_left = random.randint(0, window_number - vali_window_length)
    vali_window_right = vali_window_left + vali_window_length - 1
    # 重复距离
    overlap_distance = math.floor(1 / (1 - overlap)) - 1

    train_window = []
    vali_window = []

    for i in range(window_number):
        # 如果不是要抽取的验证窗口
        if vali_window_left - i > overlap_distance or i - vali_window_right > overlap_distance:
            train_window.append(train[i])
        elif i >= vali_window_left and i <= vali_window_right:           
            vali_window.append(train[i])
    
    return np.array(train_window), np.array(vali_window)

def train(train):
    def print_grad(name):
        print(grads[name])
        pd.DataFrame(grads[name][0].cpu().detach().numpy()).to_csv("./log/" + name + ".csv")
    losses = 0
            
    for turn in range(math.floor(train.shape[0] / batch_size)):
        optimzer.zero_grad()
        temp = train[turn]["data"].T
        batchData = np.ndarray((0,1,temp.shape[0],temp.shape[1]))
        allTarget = []
        for k in range(batch_size):
            input = train[turn*batch_size + k]["data"].T
            input = np.expand_dims(input, axis=0) 
            input = np.expand_dims(input, axis=0)
            batchData = np.concatenate((batchData,input),axis=0)
            target = [train[turn*batch_size + k]["direction"] - 1]
            allTarget = np.concatenate((allTarget,target),axis=0)
        x = torch.tensor(batchData, dtype=torch.float32, requires_grad=True)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        loss.backward()
        optimzer.step()

        # print_grad("1convA")
        # print_grad("1convB")
        # print_grad("CMA0")
        # print_grad("CMA1")
        # print_grad("CMA2")
        # print_grad("CMA3")
        # print_grad("conv0")
        # print_grad("conv1")
        # print_grad("conv2")
        # print_grad("conv3")
        # print_grad("pool0")
        print("end")

        for name, parms in myNet.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                  ' -->grad_value:', parms.grad)
        exit()
        
    return losses / (math.floor(train.shape[0] / batch_size))

def test(cv):
    losses = 0
            
    for turn in range(math.floor(cv.shape[0] / batch_size)):
        optimzer.zero_grad()
        temp = cv[turn]["data"].T
        batchData = np.ndarray((0,1,temp.shape[0],temp.shape[1]))
        allTarget = []
        for k in range(batch_size):
            input = cv[turn*batch_size + k]["data"].T
            input = np.expand_dims(input, axis=0) 
            input = np.expand_dims(input, axis=0)
            batchData = np.concatenate((batchData,input),axis=0)
            target = [cv[turn*batch_size + k]["direction"] - 1]
            allTarget = np.concatenate((allTarget,target),axis=0)
        x = torch.tensor(batchData, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        
    return losses / (math.floor(cv.shape[0] / batch_size))

def trainEpoch(data, test_data):
    min_loss = 100
    early_stop_number = 0

    for epoch in range(max_epoch):
        # 手动降低学习率，现在已废弃，改为自动学习率
        # if (epoch == 10) or (epoch == 25) or (epoch == 40):
        #     for p in optimzer.param_groups:
        #         p['lr'] *= 0.5        
        
        # 打乱非测试数据集并划分训练集和验证集
        dataset = data[0].copy()
        random.seed(0)
        train_data, cv_data = vali_split(dataset)
        # np.random.shuffle(train_data)
        
        loss_train = train(train_data)
        loss = test(cv_data)
        loss2 = test(test_data)

        # 学习率衰减
        # scheduler.step()
        scheduler.step(0.1)
        
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
        for turn in range(math.floor(test_data.shape[0] / batch_size)):
            optimzer.zero_grad()
            temp = test_data[turn]["data"].T
            batchData = np.ndarray((0,1,temp.shape[0],temp.shape[1]))
            allTarget = []
            for k in range(batch_size):
                input = test_data[turn*batch_size + k]["data"].T
                input = np.expand_dims(input, axis=0) 
                input = np.expand_dims(input, axis=0)
                batchData = np.concatenate((batchData,input),axis=0)
                target = [test_data[turn*batch_size + k]["direction"] - 1]
                allTarget = np.concatenate((allTarget,target),axis=0)
            x = torch.tensor(batchData, dtype=torch.float32)
            x = x.to(device)
            out = myNet(x)
            
            for i in range(batch_size):
                result = out[i].cpu().detach().numpy()
                result = np.expand_dims(result, axis=0)
                result = torch.from_numpy(result).to(device)
                lossL = loss_func(result, torch.tensor([0]).to(device)).cpu().detach().numpy()
                lossR = loss_func(result, torch.tensor([1]).to(device)).cpu().detach().numpy()
                if (lossL<lossR) == (allTarget[i] == 0):
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
    data = np.load(data_document + "/CNN1_"+ name + ".npy", allow_pickle=True)
    test_data = data[0] if isALLTrain and need_pretrain and not need_train else data[1]

    # 读取数据并预训练
    if need_pretrain:
        print("pretrain start!")
        basic_name = "S" + str(name_number % (people_number - 1) + 1)
        b = np.load(data_document + "/CNN1_"+ basic_name + ".npy", allow_pickle=True)
        for k in range(people_number):
            filelable = "S" + str(k+1)
            if (not isALLTrain or filelable != name) and filelable != basic_name:
                # 读取数据
                a = np.load(data_document + "/CNN1_"+ filelable + ".npy", allow_pickle=True)
                b[0] = np.hstack((a[0], b[0]))
                b[1] = np.hstack((a[1], b[1]))
        data = b
        trainEpoch(data, test_data)
        print()

    # 读取数据并训练
    if need_train:
        # 降低学习率
        if need_pretrain:
            for p in optimzer.param_groups:
                p['lr'] *= 0.1

        print("train start!")
        data = np.load(data_document + "/CNN1_"+ name + ".npy", allow_pickle=True)
        
        # # 随机选取N个数 临时起作用
        # np.random.shuffle(data[0])
        # print(data[0].size)
        # length = math.floor(data[0].size / (700 / window_length))
        # data[0] = data[0][:length]
        # print(data[0].size)
        
        trainEpoch(data, test_data)
        print()

    # 测试
    print("test start!")
    testEpoch(test_data)

if __name__=="__main__":
    myNet = myNet.to(device)
    loss_func = loss_func.to(device)
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1])
    else:
        main()