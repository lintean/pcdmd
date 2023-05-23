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

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(65),
            nn.Conv1d(65,64,3),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,2,1),
            nn.BatchNorm1d(2)
        )
        self.fcn1 = nn.Sequential(
            nn.Linear(249,200),
            nn.ELU(),
            nn.Dropout(),
            nn.BatchNorm1d(2),
        )
        self.fcn2 = nn.Sequential(
            nn.Linear(200,200),
            nn.ELU(),
            nn.Dropout(),
            nn.BatchNorm1d(2),
        )
        self.fcn3 = nn.Sequential(
            nn.Linear(200,100),
            nn.ELU(),
            nn.Dropout(),
        )
        self.fcn4 = nn.Sequential(
            nn.Linear(100,1)
        )

    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.fcn1(x) 
        x = self.fcn2(x) 
        x = self.fcn3(x) 
        x = self.fcn4(x) 
        x = x.view(-1, 2)
        output = x
        return output

def vali_split(train, vali_percent, overlap):
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

def train(train_data, batch_size): 
    losses = 0 
    for turn in range(math.ceil(train_data.shape[0] / batch_size)):
        optimzer.zero_grad()
        temp = train_data[turn]["data"]
        batchData = np.ndarray((0,temp.shape[0],temp.shape[1]))
        allTarget = []
        for k in range(batch_size):
            if turn * batch_size + k < train_data.shape[0]:
                input = train_data[turn*batch_size + k]["data"]
                input = np.expand_dims(input, axis=0) 
                batchData = np.concatenate((batchData,input),axis=0)
                target = [train_data[turn*batch_size + k]["direction"] - 1]
                allTarget = np.concatenate((allTarget,target),axis=0)
            else:
                break
        x = torch.tensor(batchData, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        loss.backward()
        optimzer.step()
    return losses / (math.ceil(train_data.shape[0] / batch_size))

def test(cv, batch_size):
    losses = 0
    for turn in range(math.ceil(cv.shape[0] / batch_size)):
        optimzer.zero_grad()
        temp = cv[turn]["data"]
        batchData = np.ndarray((0,temp.shape[0],temp.shape[1]))
        allTarget = []
        for k in range(batch_size):
            if turn * batch_size + k < cv.shape[0]:
                input = cv[turn*batch_size + k]["data"]
                input = np.expand_dims(input, axis=0)
                batchData = np.concatenate((batchData,input),axis=0)
                target = [cv[turn*batch_size + k]["direction"] - 1]
                allTarget = np.concatenate((allTarget,target),axis=0)
            else:
                break
        x = torch.tensor(batchData, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
    return losses / (math.ceil(cv.shape[0] / batch_size))

def main(name = "S2", max_epoch = "2400", batch_size = "1024", overlap = "0.5"):
    # 读取数据
    data = np.load("./data_new/CNN2_"+ name + ".npy", allow_pickle=True)

    # 训练
    batch_size = int(batch_size)
    max_epoch = int(max_epoch)
    overlap = float(overlap)
    min_loss = 100
    vali_percent = 0.1

    for epoch in range(max_epoch):        
        # 打乱非测试数据集并划分训练集和验证集
        dataset = data[0].copy()
        train_data, cv_data = vali_split(dataset, vali_percent, overlap)
        np.random.shuffle(train_data)
        
        loss_train = train(train_data, batch_size)
        loss = test(cv_data, batch_size)
        loss2 = test(data[1], batch_size)
        
        print(str(epoch) + " " + str(loss_train) + " " + str(loss) + " " + str(loss2), end="")

        if loss <= 0.09:
            break;

        print()

    # torch.save(myNet.state_dict(), './data_new/CNN2' + name + 'model.pkl')


    # 测试
    print()

    test_data = data[1]
    for num in range(10):
        t_num = 0
        f_num = 0
        for turn in range(math.ceil(test_data.shape[0] / batch_size)):
            optimzer.zero_grad()
            temp = test_data[turn]["data"]
            batchData = np.ndarray((0,temp.shape[0],temp.shape[1]))
            allTarget = []
            for k in range(batch_size):
                if turn*batch_size + k < test_data.shape[0]:
                    input = test_data[turn*batch_size + k]["data"]
                    input = np.expand_dims(input, axis=0) 
                    batchData = np.concatenate((batchData,input),axis=0)
                    target = [test_data[turn*batch_size + k]["direction"] - 1]
                    allTarget = np.concatenate((allTarget,target),axis=0)
                else:
                    break
            x = torch.tensor(batchData, dtype=torch.float32)
            x = x.to(device)
            out = myNet(x)
            
            for i in range(out.shape[0]):
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
    print(str(t_num / (t_num + f_num)))


# 设置优化器
# myNetB = CNN()
# 读取整个网络
# PATH = "./data_new/model.pth"
myNet = CNN()
optimzer = torch.optim.Adam(myNet.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.randint(0, 5)
# device = torch.device('cuda:' + str(gpu_random))
device = torch.device('cpu')
myNet=myNet.to(device)
loss_func = loss_func.to(device)

if __name__=="__main__": 
    if (len(sys.argv) > 4 and sys.argv[1].startswith("S")):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
