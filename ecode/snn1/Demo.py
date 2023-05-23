#!/usr/bin/env python
# coding: utf-8
#
# In[1]:

import sys
sys.path.append('./ecode/snn1/')
import torch
import torch.optim as optim
import numpy as np
from neural_networks import sCNN
from settings import SETTINGS
import math
from ecfg import project_root_path
import eutils.util as util
import eutils.snn_utils as sutil
from importlib import reload

# In[2]:

import torch.nn as nn
batch_size = 4
learning_rate = 3e-4
weight_decay = 3e-4
num_epoch = 1000
splited_data_document = f"D://eegdata"
# 可视化选项 列表为空表示不希望可视化
visualization_epoch = []
visualization_window_index = []


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datas):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        x, y = datas[0], datas[1]
        imgs = []
        for i in range(len(x)):  # 迭代该列表#按行循环txt文本中的内
            imgs.append((x[i], y[i], i))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        # self.transform = transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img, label, index = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        return img, label, index  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def get_logger(name, log_path):
    import logging
    reload(logging)

    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 第二步，创建一个handler，用于写入日志文件
    logfile = log_path + "/Train_" + name + ".log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def demo(name, visible=True, log_path=project_root_path + "/result/snn", q=None):
    device = torch.device(SETTINGS.training.device)
    torch.set_num_threads(1)
    logger = get_logger(name, util.makePath(log_path))
    energy_bak = []

    def test(loader, length, energy_calculate=False):
        corrects = 0
        total_loss = 0
        net.eval()
        with torch.no_grad():
            snn_energy = 0
            ann_energy = 0
            total_spike = 0
            total_all = 0
            for (inputs, labels, window_index) in loader:
                optimizer.zero_grad()
                outputs, weights = net(inputs)

                # 计算energy
                if energy_calculate:
                    for batch_index in range(batch_size):
                        temp_energy1, spike1, all1 = sutil.energy_snn_conv(data=weights[1].data[batch_index], kernel_size=weights[1].kernel_size, c_in=weights[1].c_in)
                        temp_energy2, spike2, all2 = sutil.energy_snn_fc(data=weights[2].data[batch_index], c_in=weights[2].c_in)
                        temp_energy3, spike3, all3 = sutil.energy_snn_fc(data=weights[3].data[batch_index], c_in=weights[3].c_in)
                        temp_energy4 = sutil.energy_ann_fc(c_in=weights[4].c_in, c_out=weights[4].c_out)
                        snn_energy += temp_energy1 + temp_energy2 + temp_energy3 + temp_energy4
                        total_spike += spike1 + spike2 + spike3
                        total_all += all1 + all2 + all3

                        feature_size = weights[1].data.shape[-1] * weights[1].data.shape[-2]
                        ann_energy += sutil.energy_ann_conv(kernel_size=weights[1].kernel_size, feature_size=feature_size,  c_in=weights[1].c_in, c_out=weights[1].c_out)
                        ann_energy += sutil.energy_ann_fc(c_in=weights[2].c_in, c_out=weights[2].c_out)
                        ann_energy += sutil.energy_ann_fc(c_in=weights[3].c_in, c_out=weights[3].c_out)
                        ann_energy += sutil.energy_ann_fc(c_in=weights[4].c_in, c_out=weights[4].c_out)

                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                corrects += (predicted == labels).cpu().sum()
                total_loss += loss.item()

            if energy_calculate:
                logger.info("snn_energy: " + str(snn_energy) + " ann_energy: " + str(ann_energy) + " averageR: " + str(total_spike / total_all))
                energy_bak.append([snn_energy, ann_energy, total_spike, total_all, total_spike/total_all])

        acc = corrects.cpu().numpy() / length
        return acc, total_loss / length

    data = np.load(splited_data_document + '/snn/export_1s_person_' + name + '.npz')
    x = data['data']
    y = data['label']
    index = [i for i in range(y.shape[0])]
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    val_line = math.floor(len(x) * 0.6)
    test_line = math.floor(len(x) * 0.8)
    train_x = torch.from_numpy(x[:val_line]).float().unsqueeze(1).to(device)
    train_y = (torch.from_numpy(y[:val_line]).squeeze(1).long()-1).to(device)
    val_x = torch.from_numpy(x[val_line:test_line]).float().unsqueeze(1).to(device)
    val_y = (torch.from_numpy(y[val_line:test_line]).squeeze(1).long()-1).to(device)
    test_x = torch.from_numpy(x[test_line:]).float().unsqueeze(1).to(device)
    test_y = (torch.from_numpy(y[test_line:]).squeeze(1).long()-1).to(device)

    train_loader = torch.utils.data.DataLoader(dataset=MyDataset([train_x, train_y]), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=MyDataset([val_x, val_y]), batch_size=batch_size, shuffle=False,
                                                      num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=MyDataset([test_x, test_y]), batch_size=batch_size, shuffle=False,
                                                      num_workers=0, drop_last=True)

    T = 10
    len_out = int(max(train_y))+1

    net = sCNN(len_out, T).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    logger.info('Number of model parameters is {}'.format(sum(p.numel() for p in net.parameters())))
    running_loss = 0.0
    running_corrects = 0
    total_utt = 0

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=1e-5, threshold_mode='rel', cooldown=10,
        min_lr=0, eps=1e-8)

    best_val_acc = 0
    best_acc = 0
    best_acc_epoch = 0
    best_weight = None
    for epoch in range(num_epoch):
        net.train()
        for (train_fingerprints, train_ground_truth, window_index) in train_loader:
            optimizer.zero_grad()
            net.zero_grad()
            inputs = train_fingerprints
            labels = train_ground_truth
            outputs, weights = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = 0, 0, 0, 0, 0, 0
        val_acc, val_loss = test(val_loader, len(val_x))
        # scheduler.step(0.1)

        # 测试
        if visible:
            train_acc, train_loss = test(train_loader, len(train_x))
            test_acc, test_loss = test(test_loader, len(test_x), energy_calculate=False)

            logger.info(str(epoch) + ' epoch ,loss is: ' + str(train_loss) + " " + str(val_loss) + " " + str(test_loss))
            logger.info(str(train_acc) + " " + str(val_acc) + " " + str(test_acc))
            logger.info("")

            # weight = net.state_dict()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_acc = test_acc
                best_acc_epoch = epoch
                # best_weight = copy.deepcopy(weight)
            if epoch == num_epoch - 1:
                logger.info("S" + name + ": " + str(test_acc))
        else:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc, test_loss = test(test_loader, len(test_x))
                best_acc = test_acc
                best_acc_epoch = epoch
                if epoch == num_epoch - 1:
                    logger.info("S" + name + ": " + str(test_acc))


    # net_params =  sum(p.numel() for p in net.parameters())
    logger.info("S" + name + ": " + str(best_acc) + " epoch: " + str(best_acc_epoch))

    if q != None:
        q.put(np.array([name, best_acc, best_acc_epoch, energy_bak[best_acc_epoch][0], energy_bak[best_acc_epoch][1],
                        energy_bak[best_acc_epoch][2], energy_bak[best_acc_epoch][3], energy_bak[best_acc_epoch][4]]))

if __name__ == "__main__":
    demo("1")


