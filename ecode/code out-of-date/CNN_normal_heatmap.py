from scipy.signal import hilbert
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torch.autograd import Function
from torch.autograd import Variable
import math
import random
import sys
from parameters import *
import seaborn as sns


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


def heatmap(_, epoch, number, matrix_num, title='crossmodal attention matrix', x_label='No.', y_label='No.'):
    #     print(type(_[0]))
    #     print(_[0].cpu().detach().numpy().shape)
    a = _[0].cpu().detach().numpy()
    #     print(a.shape)
    fig, ax = plt.subplots(figsize=(20, 20))
    heatmap = sns.heatmap(pd.DataFrame(np.round(a, 2)), xticklabels=True, yticklabels=True, square=True)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel(x_label, fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的

    fig.savefig("./picture/" + names[0] + "/" + str(epoch) + "_" + str(number) + "_" + str(matrix_num) + ".png")
    plt.close(fig)

def isLeftPredicted(result):
    result = result.cpu().detach().numpy()
    result = np.expand_dims(result, axis=0)
    result = torch.from_numpy(result).to(device)
    lossL = loss_func(result, torch.tensor([0]).to(device)).cpu().detach().numpy()
    lossR = loss_func(result, torch.tensor([1]).to(device)).cpu().detach().numpy()
    return lossL < lossR


def train(train):
    losses = 0

    for turn in range(math.floor(train.shape[0] / batch_size)):
        optimzer.zero_grad()
        temp = train[turn]["data"].T
        batchData = np.ndarray((0, 1, temp.shape[0], temp.shape[1]))
        allTarget = []
        for k in range(batch_size):
            input = train[turn * batch_size + k]["data"].T
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=0)
            batchData = np.concatenate((batchData, input), axis=0)
            target = [train[turn * batch_size + k]["direction"] - 1]
            allTarget = np.concatenate((allTarget, target), axis=0)
        x = torch.tensor(batchData, dtype=torch.float32)
        x = x.to(device)
        out = myNet(x)

        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()
        loss.backward()
        optimzer.step()
    # scheduler.step()
    scheduler.step(metrics=0.1)

    return losses / (math.floor(train.shape[0] / batch_size))


def test(cv, need_weight=False, targetIndex=[], label="", name=""):
    losses = 0
    max_list = np.zeros((16,20), dtype = np.int)
    # output_list = [True for i in range(math.floor(cv.shape[0] / batch_size))]

    for turn in range(math.floor(cv.shape[0] / batch_size)):
        optimzer.zero_grad()
        temp = cv[turn]["data"].T
        batchData = np.ndarray((0, 1, temp.shape[0], temp.shape[1]))
        allTarget = []
        for k in range(batch_size):
            input = cv[turn * batch_size + k]["data"].T
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=0)
            batchData = np.concatenate((batchData, input), axis=0)
            target = [cv[turn * batch_size + k]["direction"] - 1]
            allTarget = np.concatenate((allTarget, target), axis=0)
        x = torch.tensor(batchData, dtype=torch.float32)
        x = x.to(device)

        if need_weight:
            out, weightA, weightB, max_index, wA, wB = myNet(x, need_weight=need_weight)
            trailIndex = cv[turn]["trail"]
            max_list[max_index, trailIndex] += 1

            for i in range(batch_size):
                if turn in targetIndex:
                    # itemIndex = train[turn * batch_size + i]["index"]
                    isLeft = isLeftPredicted(out[i])
                    isAAttend = allTarget[i] == 0
                    # output_list[turn] = False if isLeft != isAAttend else True

                    attend_str = " attend=A" if isAAttend else " attend=A"
                    predict_str = " predict=A" if isLeft else " predict=B "
                    title_str = attend_str + predict_str

                    # heatmap(wA, "test_" + label, turn, "wA", title="wA" + title_str + str(max_index))
                    # heatmap(wB, "test_" + label, turn, "wB", title="wB" + title_str + str(max_index))
        else:
            out = myNet(x, need_weight=need_weight)

        loss = loss_func(out, torch.tensor(allTarget, dtype=torch.long).to(device))
        losses = losses + loss.cpu().detach().numpy()

    if need_weight:
        pd.DataFrame(max_list).to_csv("./result/" + name + "max_list.csv")
        # pd.DataFrame(output_list).to_csv("./result/" + name + "output_list.csv")

    return losses / (math.floor(cv.shape[0] / batch_size))

def trainEpoch(data, test_data, name):
    min_loss = 100
    early_stop_number = 0

    targetIndex = np.random.choice(a=math.floor(data[0].shape[0] * 0.9), size=10, replace=False)
    print(targetIndex)

    test(np.concatenate((data[0], data[1]), axis=0), need_weight=True, targetIndex=targetIndex, label="start", name=name)

    for epoch in range(max_epoch):

        # 打乱非测试数据集并划分训练集和验证集
        dataset = data[0].copy()
        train_data, cv_data = vali_split(dataset)
        np.random.shuffle(train_data)

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

    # 可视化
    # print(data[0].shape)
    # print(data[1].shape)
    # print(np.concatenate((data[0], data[1]),axis=0).shape)
    test(np.concatenate((data[0], data[1]),axis=0), need_weight=True, targetIndex=targetIndex, label="end", name=name)


def testEpoch(test_data):
    total_t_num = 0
    total_f_num = 0
    for num in range(10):
        t_num = 0
        f_num = 0
        for turn in range(math.floor(test_data.shape[0] / batch_size)):
            optimzer.zero_grad()
            temp = test_data[turn]["data"].T
            batchData = np.ndarray((0, 1, temp.shape[0], temp.shape[1]))
            allTarget = []
            for k in range(batch_size):
                input = test_data[turn * batch_size + k]["data"].T
                input = np.expand_dims(input, axis=0)
                input = np.expand_dims(input, axis=0)
                batchData = np.concatenate((batchData, input), axis=0)
                target = [test_data[turn * batch_size + k]["direction"] - 1]
                allTarget = np.concatenate((allTarget, target), axis=0)
            x = torch.tensor(batchData, dtype=torch.float32)
            x = x.to(device)
            out = myNet(x)

            for i in range(batch_size):
                ifLeft = isLeftPredicted(out[i])
                if ifLeft == (allTarget[i] == 0):
                    t_num = t_num + 1
                else:
                    f_num = f_num + 1

        print(str(t_num) + " " + str(f_num))
        total_t_num = total_t_num + t_num
        total_f_num = total_f_num + f_num
    print(str(total_t_num / (total_t_num + total_f_num)))


def main(name="S2", data_document="./data_new"):
    # 参数init
    name_number = int(name[1:])

    # 先读取测试数据
    data = np.load("./" + data_document + "/CNN1_" + name + ".npy", allow_pickle=True)
    test_data = data[0] if isALLTrain and need_pretrain and not need_train else data[1]

    # 读取数据并预训练
    if need_pretrain:
        print("pretrain start!")
        basic_name = "S" + str(name_number % (people_number - 1) + 1)
        b = np.load("./" + data_document + "/CNN1_" + basic_name + ".npy", allow_pickle=True)
        for k in range(people_number):
            filelable = "S" + str(k + 1)
            if (not isALLTrain or filelable != name) and filelable != basic_name:
                # 读取数据
                a = np.load("./" + data_document + "/CNN1_" + filelable + ".npy", allow_pickle=True)
                b[0] = np.hstack((a[0], b[0]))
                b[1] = np.hstack((a[1], b[1]))
        data = b
        trainEpoch(data, test_data, name)
        print()

    # 读取数据并训练
    if need_train:
        # 降低学习率
        if need_pretrain:
            for p in optimzer.param_groups:
                p['lr'] *= 0.1

        print("train start!")
        data = np.load("./" + data_document + "/CNN1_" + name + ".npy", allow_pickle=True)

        # # 随机选取N个数 临时起作用
        # np.random.shuffle(data[0])
        # print(data[0].size)
        # length = math.floor(data[0].size / (700 / window_length))
        # data[0] = data[0][:length]
        # print(data[0].size)

        trainEpoch(data, test_data, name)
        print()

    # 测试
    print("test start!")
    testEpoch(test_data)

if __name__ == "__main__":
    myNet = myNet.to(device)
    loss_func = loss_func.to(device)
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1], sys.argv[2])
    else:
        main()