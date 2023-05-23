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
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,10,(66,9)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fcn = nn.Sequential(
            nn.Linear(10,10),nn.Sigmoid(),nn.Dropout(p = 0.38),
            nn.Linear(10,2),nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)  
        x = x.view(-1, 10)
        output = self.fcn(x)
        return output

def train(train): 
    losses = 0
    for turn in range(train.shape[0]):    
        optimzer.zero_grad() 
        input = train[turn]["data"].T
        input = input[np.newaxis, :]
        input = input[np.newaxis, :]
        x = torch.tensor(input, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor([train[turn]["direction"] - 1]).to(device))
        losses = losses + loss.cpu().detach().numpy()
        loss.backward()
        optimzer.step()
    return losses / train.shape[0]

def test(cv):
    losses = 0
    for turn in range(cv.shape[0]):
        optimzer.zero_grad() 
        input = cv[turn]["data"].T
        input = input[np.newaxis, :]
        input = input[np.newaxis, :]
        x = torch.tensor(input, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor([cv[turn]["direction"] - 1]).to(device))
        losses = losses + loss.cpu().detach().numpy()
    return losses / cv.shape[0]

def main(name = "S2"):
    # 读取数据
    data = np.load("./data_new/CNN1_"+ name + ".npy", allow_pickle=True)

    # 训练
    bais = 0
    min_loss = 100
    early_stop_number = 0
    max_epoch = 600

    for epoch in range(max_epoch):
        if (epoch == 10) or (epoch == 25) or (epoch == 40):
            for p in optimzer.param_groups:
                p['lr'] *= 0.5
        
        # 打乱非测试数据集并划分训练集和验证集
        np.random.shuffle(data[0])
        a = math.floor(data[0].shape[0] * 0.9)
        train_data = data[0][:a].copy()
        cv_data = data[0][a+1:].copy()
        
        loss_train = train(train_data)
        loss = test(cv_data)
        loss2 = test(data[1])
        
        print(str(epoch) + " " + str(loss_train) + " " + str(loss) + " " + str(loss2), end="")

        if loss > min_loss:
            early_stop_number = early_stop_number + 1
        else:
            early_stop_number = 0
            min_loss = loss

        print(" early_stop_number: ", end="")
        print(early_stop_number, end="")
        print()
    #     if early_stop_number >= 10:
    #         break;

    # torch.save(myNet.state_dict(), './model/cnn/' + name + '/model.pkl')


    # 测试
    print()

    test_data = data[1]
    for k in range(10):
        t_num = 0
        f_num = 0
        for turn in range(test_data.shape[0]):
            input = test_data[turn]["data"].T
            input = input[np.newaxis, :]
            input = input[np.newaxis, :]
            x = torch.tensor(input, dtype=torch.float32)
            x = x.to(device)
            out = myNet(x)
            lossL = loss_func(out, torch.tensor([0]).to(device)).cpu().detach().numpy()
            lossR = loss_func(out, torch.tensor([1]).to(device)).cpu().detach().numpy()
    #         print(str(lossL) + " " + str(lossR))
            if (lossL<lossR) == (test_data[turn]["direction"] - 1 == 0):
                t_num = t_num + 1
            else:
                f_num = f_num + 1

        print(str(t_num) + " " + str(f_num))
    print(str(t_num / (t_num + f_num)))


# 设置优化器
myNet = CNN()
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)
# optimzer = torch.optim.Adam(myNet.parameters(), lr=0.01, weight_decay=0.0000001)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.randint(0, 7)
device = torch.device('cuda:' + str(gpu_random))
# device = torch.device('cpu')
myNet=myNet.to(device)
loss_func = loss_func.to(device)

if __name__=="__main__": 
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1])
    else:
        main()
