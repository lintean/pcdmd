from STDP_extractor import STDP_extractor
import numpy as np
import os
import _pickle as pkl
import torch
import cv2
import time
Tmax=1
dt=.02
C_l=0.0005
spike_thres = [0,0,0]
V1_kernel_sizes=[3,5,7]
V1_feature_maps=[15,40,50]
V1_thres=[7,15,30]
V1_pools=[2,2,6]

chain_loss_record1=[]
chain_loss_record2=[]
"""
2.多尺度深度脉冲特征提取算法
    a.使用3x3,5x5,7x7卷积，池化层都使用2x2，异步训练，最后contact
    b.用三种尺寸，每一种尺寸对应一组5X5的卷积核，最后contact
"""
model = STDP_extractor(0.004,-0.003)
weight_save_root_path=r'./weight'

def write_pkl(path,data):
  with open(path,'wb') as f:
      pkl.dump(data,f)
      f.flush()
      f.close()

def read(path):
    with open(path, 'rb') as f:
        s= pkl.load(f)
        f.close()
    return s

def train_v1(train_loader):
    W1_name = 'v1_0_1.pkl'
    if not os.path.exists(os.path.join(weight_save_root_path, W1_name)):
        W1 = np.random.normal(0.8, 0.05, [V1_feature_maps[0], 2, V1_kernel_sizes[0], V1_kernel_sizes[0]])
    else:
        W1 = read(os.path.join(weight_save_root_path, W1_name))
    flag=0
    m = model.measure(W1)
    if m >= C_l:
        max_epoches = 10000
        for epoch in range(1, max_epoches + 1):
            i = 0
            for (images, labels) in train_loader:
                input = images[0][0].numpy()
                i += 1
                temporal_coding = model.temporal_dog_coding(input,Tmax,spike_thres[0])
                # print('编码层正确')

                temporal_coding = torch.Tensor(temporal_coding).permute(2,0,1)

                conv_time1, conv_V1 = model.spike_conv(temporal_coding,W1,V1_thres[0],Tmax,dt)
                # print('卷积层正确')
                W1 = model.STDP(temporal_coding, conv_time1, conv_V1, W1)
                # print('正确训练')
                m = model.measure(W1)
                chain_loss_record1.append (m)
                print('V1 串行 0 层,epoch ' + str(epoch) + ', 第', str(i), '次训练loss = ', str(m))
                if m < 0:
                    print('学习率过大，请调小')
                    exit(0)
                if epoch > 1:
                    if m < C_l:
                        # 说明收敛
                        print('V1 串行 0 层,epoch ' + str(epoch) + ', 第', str(i), '次训练  '+'收敛，停止运行')
                        write_pkl(os.path.join(weight_save_root_path, W1_name), W1)
                        #return
                        flag=1
                        break
            if flag:
               break 
    W2_name = 'v1_1_1.pkl'
    if not os.path.exists (os.path.join (weight_save_root_path, W2_name)):
        W2 = np.random.normal (0.8, 0.05,
                               [V1_feature_maps[1], V1_feature_maps[0], V1_kernel_sizes[1], V1_kernel_sizes[1]])
    else:
        W2 = read (os.path.join (weight_save_root_path, W2_name))

    m = model.measure (W2)

    if m >= C_l:
        max_epoches = 10000
        for epoch in range (1, max_epoches + 1):
            i = 0
            for (images, labels) in train_loader:
                input = images[0][0].numpy ()
                i += 1
                temporal_coding = model.temporal_dog_coding (input, Tmax, spike_thres[0])
                # print('编码层正确')

                temporal_coding = torch.Tensor (temporal_coding).permute (2, 0, 1)

                conv_time1, conv_V1 = model.spike_conv (temporal_coding, W1, V1_thres[0], Tmax, dt)
                pool_time1 = model.spike_pooling (conv_time1, V1_pools[0], V1_pools[0])
                conv_time2, conv_V2 = model.spike_conv (pool_time1, W2, V1_thres[1], Tmax, dt)
                # print('卷积层正确')
                W2 = model.STDP (pool_time1, conv_time2, conv_V2, W2)
                # print('正确训练')
                m = model.measure (W2)
                chain_loss_record2.append (m)
                print ('V1 串行 1 层,epoch ' + str (epoch) + ', 第', str (i), '次训练loss = ', str (m))
                if m < 0:
                    print ('学习率过大，请调小')
                    exit (0)
                if epoch > 1:
                    if m < C_l:
                        # 说明收敛
                        print ('V1 串行 1 层,epoch ' + str (epoch) + ', 第', str (i),
                               '次训练  ' + '收敛，停止运行')
                        write_pkl (os.path.join (weight_save_root_path, W2_name), W2)
                        write_pkl ('MNIST_chain_loss_record1.pkl', chain_loss_record1)
                        write_pkl ('MNIST_chain_loss_record2.pkl', chain_loss_record2)
                        print ('finish !!!!!')
                        # return
    W2_name = 'v1_2_1.pkl'
    if not os.path.exists (os.path.join (weight_save_root_path, W2_name)):
        W2 = np.random.normal (0.8, 0.05,
                               [V1_feature_maps[1], V1_feature_maps[0], V1_kernel_sizes[1], V1_kernel_sizes[1]])
    else:
        W2 = read (os.path.join (weight_save_root_path, W2_name))

    m = model.measure (W2)

    if m >= C_l:
        max_epoches = 10000
        for epoch in range (1, max_epoches + 1):
            i = 0
            for (images, labels) in train_loader:
                input = images[0][0].numpy ()
                i += 1
                temporal_coding = model.temporal_dog_coding (input, Tmax, spike_thres[0])
                # print('编码层正确')

                temporal_coding = torch.Tensor (temporal_coding).permute (2, 0, 1)

                conv_time1, conv_V1 = model.spike_conv (temporal_coding, W1, V1_thres[0], Tmax, dt)
                pool_time1 = model.spike_pooling (conv_time1, V1_pools[0], V1_pools[0])
                conv_time2, conv_V2 = model.spike_conv (pool_time1, W2, V1_thres[1], Tmax, dt)
                # print('卷积层正确')
                W2 = model.STDP (pool_time1, conv_time2, conv_V2, W2)
                # print('正确训练')
                m = model.measure (W2)
                chain_loss_record2.append (m)
                print ('V1 串行 1 层,epoch ' + str (epoch) + ', 第', str (i), '次训练loss = ', str (m))
                if m < 0:
                    print ('学习率过大，请调小')
                    exit (0)
                if epoch > 1:
                    if m < C_l:
                        # 说明收敛
                        print ('V1 串行 1 层,epoch ' + str (epoch) + ', 第', str (i),
                               '次训练  ' + '收敛，停止运行')
                        write_pkl (os.path.join (weight_save_root_path, W2_name), W2)
                        write_pkl ('MNIST_chain_loss_record1.pkl', chain_loss_record1)
                        # write_pkl ('MNIST_chain_loss_record2.pkl', chain_loss_record2)
                        print ('finish !!!!!')
                        return


# 同尺寸，同层不同尺寸卷积核
class SCNN_Extractor_V1(object):
    def __init__(self):
        self.train_monitor = None

    def train(self,data_loader):
        start = time.time()
        train_v1(data_loader)
        print('串行结构训练耗时：',time.time()-start,' s')

    def extractor(self,train_loader,test_loader):
        W1 = read(os.path.join(weight_save_root_path, 'v1_0_1.pkl'))
        W2 = read(os.path.join(weight_save_root_path, 'v1_1_1.pkl'))
        accum_time = 0
        for i,(images , labels) in enumerate(train_loader):
            print('train',i)
            start =time.time()
            input = images[0][0].numpy()
            temporal_coding = model.temporal_dog_coding (input, Tmax, spike_thres[0])
            temporal_coding = torch.Tensor (temporal_coding).permute (2, 0, 1)
            conv_time1, conv_V1 = model.spike_conv (temporal_coding, W1, V1_thres[0], Tmax, dt)
            pool_time1 = model.spike_pooling (conv_time1, V1_pools[0], V1_pools[0])
            conv_time2, conv_V2 = model.spike_conv (pool_time1, W2, V1_thres[1], Tmax, dt)
            pool_time2 = model.spike_pooling (conv_time2, V1_pools[1], V1_pools[1])
            end = time.time()
            accum_time+=(end-start)
        for i,(images , labels) in enumerate(test_loader):
            print('train',i)
            start =time.time()
            input = images[0][0].numpy()
            temporal_coding = model.temporal_dog_coding (input, Tmax, spike_thres[0])
            temporal_coding = torch.Tensor (temporal_coding).permute (2, 0, 1)
            conv_time1, conv_V1 = model.spike_conv (temporal_coding, W1, V1_thres[0], Tmax, dt)
            pool_time1 = model.spike_pooling (conv_time1, V1_pools[0], V1_pools[0])
            conv_time2, conv_V2 = model.spike_conv (pool_time1, W2, V1_thres[1], Tmax, dt)
            pool_time2 = model.spike_pooling (conv_time2, V1_pools[1], V1_pools[1])
            end = time.time()
            accum_time+=(end-start)
        print('串行结构每条数据前馈时间：',accum_time/(len(train_loader)+len(test_loader)),'s')
