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
            nn.Conv2d(3,10,(66,9)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fcn = nn.Sequential(
            nn.Linear(10,10),nn.Sigmoid(),
            nn.Linear(10,2),nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)  
#         print(x.shape)
        x = x.view(-1, 10)
        output = self.fcn(x)
        return output

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)

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

def change_data_alpha(input):
    channel_number = 64
    
    output = np.ndarray((0,3,channel_number + 2,input.shape[3]))
    for k in range(input.shape[0]):
        channel_set = np.ndarray((1,0,channel_number + 2,input.shape[3]))
#         print(pd.DataFrame(input[k][0]))

        audio = input[k][0][:2,:]
        for j in range(math.floor(input.shape[2] / channel_number)):
            left = 2 + j * channel_number
            right = left + channel_number
            temp = input[k][0][left:right, :]
            temp = np.concatenate((audio, temp), axis=0)
            temp = np.expand_dims(temp, axis=0) 
            temp = np.expand_dims(temp, axis=0)
            channel_set = np.concatenate((channel_set, temp), axis=1)
            
#         print(pd.DataFrame(channel_set[0][2]))
#         sys.exit()
        output = np.concatenate((output, channel_set), axis=0)
    
    return output

def train(train, batch_size): 
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
            
        # 转换alpha数据
        batchData = change_data_alpha(batchData)
        x = torch.tensor(batchData, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        loss.backward()
        optimzer.step()
        
    return losses / (math.floor(train.shape[0] / batch_size))

def test(cv, batch_size):
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
            
        # 转换alpha数据
        batchData = change_data_alpha(batchData)
        x = torch.tensor(batchData, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)
        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        
    return losses / (math.floor(cv.shape[0] / batch_size))

def trainEpoch(data, batch_size, max_epoch, overlap):
    min_loss = 100
    early_stop_number = 0
    vali_percent = 0.1

    for epoch in range(max_epoch):
        if (epoch == 10) or (epoch == 25) or (epoch == 40):
            for p in optimzer.param_groups:
                p['lr'] *= 0.5
        
        # 打乱非测试数据集并划分训练集和验证集
        dataset = data[0].copy()
        train_data, cv_data = vali_split(dataset, vali_percent, overlap)
        np.random.shuffle(train_data)
        
        loss_train = train(train_data, batch_size)
        loss = test(cv_data, batch_size)
        loss2 = test(data[1], batch_size)
        
        print(str(epoch) + " " + str(loss_train) + " " + str(loss) + " " + str(loss2), end="")

        if loss > min_loss:
            early_stop_number = early_stop_number + 1
        else:
            early_stop_number = 0
            min_loss = loss

        print(" early_stop_number: ", end="")
        print(early_stop_number, end="")
        print()

#         if early_stop_number >= 10:
#             break

    # torch.save(myNet.state_dict(), './data_new/CNN1' + name + 'model.pkl')

def testEpoch(test_data, batch_size):
    print()

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
                
            # 转换alpha数据
            batchData = change_data_alpha(batchData)
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
    print(str(t_num / (t_num + f_num)))


def main(name = "S1", max_epoch = "100", batch_size = "1", overlap = "0.5", need_pretrain = "False"):
    # 参数init
    batch_size = int(batch_size)
    max_epoch = int(max_epoch)
    overlap = float(overlap)
    need_pretrain = need_pretrain == "True"
    name_number = int(name[1:])
    people_number = 18

    # 读取数据并预训练
    if need_pretrain:
        print("pretarin start!")
        basic_name = "S" + str((name_number + 1) % people_number)
        b = np.load("./data_new/CNN1_"+ basic_name + ".npy", allow_pickle=True)
        for k in range(people_number):
            filelable = "S" + str(k+1)
            if filelable != name and filelable != basic_name:
                # 读取数据
                a = np.load("./data_new/CNN1_"+ filelable + ".npy", allow_pickle=True)
                b[0] = np.hstack((a[0], b[0]))
                b[1] = np.hstack((a[1], b[1]))
        data = b
        trainEpoch(data, batch_size, max_epoch, overlap)
        print()

    # 读取数据并训练
    print("tarin start!")
    data = np.load("./data_new/CNN1_"+ name + ".npy", allow_pickle=True)
    trainEpoch(data, batch_size, max_epoch, overlap)

    # 测试
    testEpoch(data[1], batch_size)

# 设置优化器
# myNetB = CNN()
# 读取整个网络
# PATH = "./data_new/model.pth"
myNet = CNN()
myNet.apply(weights_init_uniform)
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.1, weight_decay=0.0000001)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
gpu_random = random.randint(0, 7)
# device = torch.device('cuda:' + str(gpu_random))
device = torch.device('cpu')
myNet=myNet.to(device)
loss_func = loss_func.to(device)

if __name__=="__main__": 
    if (len(sys.argv) > 5 and sys.argv[1].startswith("S")):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
