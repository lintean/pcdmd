# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: zzx

Python 3.5.2

"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
import _pickle as pkl
# from torchviz import make_dot, make_dot_from_trace
# from spiking_model_new import *
# from Spike_model_ZhangZhixuan_New import*
# from Spike_mode_ZhangZhixuan import*
# from spiking_model import *
# from MSSNN import *
from Caltech_SpikeClassifier import*
import scipy.io as io
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

names = 'spiking_model'
data_path = './raw/' #todo: input your data path


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device =torch.device('cpu')
# train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#
# test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
def read(path):
    with open(path, 'rb') as f:
        s= pkl.load(f)
        f.close()
    return s

# 使用自己编码好的数据集
# datas = io.loadmat('codedMNIST.mat')
# train_datas = torch.tensor(datas['codedTrainData'].T).float()/3
# train_labels = torch.tensor(datas['trainlabel']).long().squeeze(1)
# test_datas = torch.tensor(datas['codedTestData'].T).float()/3
# test_labels = torch.tensor(datas['testlabel']).long().squeeze(1)
# train_datas[train_datas==float('inf')] = nonspike
# test_datas[test_datas==float('inf')] = nonspike
train_dataset = read('caltech_classify_train_data.pkl')

train_x = torch.tensor(train_dataset[0]).float()
train_y = torch.tensor(train_dataset[1]).long()
train_dataset = [train_x,train_y]

test_dataset = read('caltech_classify_test_data.pkl')
test_x = torch.tensor(test_dataset[0]).float()
test_y = torch.tensor(test_dataset[1]).long()
test_dataset = [test_x,test_y]

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datas):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        x,y = datas[0],datas[1]
        imgs = []
        for i in range(len(x)):  # 迭代该列表#按行循环txt文本中的内
            imgs.append(( x[i] , y[i]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        # self.transform = transform


    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # img = self.loader(fn)  # 按照路径读取图片
        # if self.transform is not None:
        #     img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

train_loader = torch.utils.data.DataLoader(dataset=MyDataset(train_dataset), batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=MyDataset(test_dataset), batch_size=batch_size, shuffle=False, num_workers=0)



snn = SpikeMLP()

# 根据ANN 初始化参数
# ANN_dict = torch.load('./checkpoint/mlp784_800_10_500.pth')
# SNN_dict = snn.state_dict()
# # 1. filter out unnecessary keys
# # pretrained_dict = {
# # k.replace('convnet.' + str(int(k.split('.')[1])) + '.', "conv" + str(int(k.split('.')[1]) // 3 + 1) + '_').replace(
# #     'fc.' + str(int(k.split('.')[1])) + '.', "fc" + str(int(k.split('.')[1]) // 2 + 1) + '_'): v for k, v in
# # ANN_dict.items() if
# # k.replace('convnet.' + str(int(k.split('.')[1])) + '.', "conv" + str(int(k.split('.')[1]) // 3 + 1) + '_').replace(
# #     'fc.' + str(int(k.split('.')[1])) + '.', "fc" + str(int(k.split('.')[1]) // 2 + 1) + '_') in SNN_dict}
# pretrained_dict = {k.replace('.0.' ,'_') : v for k, v in ANN_dict.items() if k.replace('.0.' ,'_') in SNN_dict}
# # pretrained_dict = {k.replace('.0.' ,'_').replace('.' ,'_') : v for k, v in ANN_dict.items() if k.replace('.0.' ,'_').replace('.' ,'_') in SNN_dict}
# # 2. overwrite entries in the existing state dict
# SNN_dict.update(pretrained_dict)
# snn.load_state_dict(SNN_dict)

snn.to(device)
# images = torch.ones([1,1,28,28]).float().to(device)
# make_dot(snn(images), params=dict(list(snn.named_parameters()) + [('images', images)])).view()


criterion = loss_fun.apply
# optimizer = torch.optim.SGD(snn.parameters(), lr=learning_rate)
optimizer =torch.optim.Adam(snn.parameters(),lr=learning_rate)


# import scipy.io as io
# datas = io.loadmat('codedMNIST.mat')
# train_datas = torch.tensor(datas['codedTrainData'].T).float()/3
# train_labels = torch.tensor(datas['trainlabel']).long()
# test_datas = torch.tensor(datas['codedTestData'].T).float()/3
# test_labels = torch.tensor(datas['testlabel']).long()
# train_datas[train_datas==float('inf')] = nonspike
# test_datas[test_datas==float('inf')] = nonspike
# train_datas = train_datas.unsqueeze(dim=1)
# test_datas = test_datas.unsqueeze(dim=1)
# def write_out(x,lr):
#     file ='record_lr'+str(lr)+'.txt'
#     x+='\n'
#     with open(file,'a+') as f:
#         f.write(x)
#         f.flush()
#         f.close()

# for sigma in sigmas:
acc_record = list([])
train_acc = list([])
for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.clock()
        train_correct = 0
        train_total = 0
        for i,(images, labels) in enumerate(train_loader):
            snn.zero_grad()
            images += torch.zeros_like(images).normal_(0, sigma)
            images = images.clamp(min=0,max=Tmax)
            images = images.float().to(device)
            labels_ = torch.zeros(images.shape[0], 2).scatter_(1, labels.view(-1, 1), 1)
            outputs = snn(images)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().min(1)
            train_total += float(labels.size(0))
            train_correct += float(predicted.eq(labels).sum().item())

            loss.backward()
            optimizer.step()
            running_loss += loss

            if (i+1)%1== 0:
                  print ('Epoch [%d/%d], Step [%d/%d], Loss: %f'
                         %(epoch+1, num_epochs, (i+1), len(train_dataset)//batch_size,running_loss ))#len(train_dataset)
                  running_loss = 0
                  print('Time elasped:', time.time()-start_time)
        print('Total %d epochs ,learing rate %f ,sigma %f,Train Accuracy of the model of %d Epoch on the 60000 train images: %f' % (num_epochs,learning_rate,sigma,epoch+1,100. * float(train_correct) / float(train_total)))
        train_acc.append(100. * float(train_correct) / float(train_total))
        correct = 0
        total = 0
        #optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

        with torch.no_grad():
            for batch_idx,(inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                snn.zero_grad()
                outputs = snn(inputs)
                _, predicted = outputs.cpu().min(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
        print('Total %d epochs ,learing rate %f ,sigma %f,Test Accuracy of the model of %d Epoch on the 10000 test images: %f' % (num_epochs,learning_rate,sigma,epoch+1,100 * correct / total))
        acc = 100. * float(correct) / float(total)
        end_time = time.clock()
        print('Time consuming of %d Epoch is : %f\n'%(epoch+1,end_time-start_time))
        acc_record.append(acc)
        if (epoch+1)==num_epochs:
            torch.save(snn.state_dict(), 'Adam_MLP_Tmax%d_lr%f_jitter%f_E%f.pth'%(Tmax,learning_rate,sigma,loss_thres) )
print('In learing rate %f and sigma %f , the highest accuracy of training : %f' % (learning_rate, sigma, max(train_acc)))
print('In learing rate %f and sigma %f , the highest accuracy of testing :%f '%(learning_rate,sigma, max(acc_record)))
#     recording = 'In learing rate %f and sigma %f,the highest accuracy of training: %f ,the highest accuracy of testing: %f\n'% (learning_rate, sigma, max(train_acc),max(acc_record))
#     write_out(recording,learning_rate)
#     print('In learing rate %f and sigma %f , The outcome is saved!!!!! ' % ( learning_rate, sigma))
# print('\n\n\nfinish!!!')
