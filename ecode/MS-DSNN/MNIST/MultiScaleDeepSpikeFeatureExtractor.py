from STDP_extractor import STDP_extractor
import numpy as np
import os
import _pickle as pkl
import torch
import cv2
Tmax=1
dt=.02
C_l=0.0005
spike_thres = [0.1,0.2,0.4]
V1_kernel_sizes=[3,5,7]
V1_feature_maps=[15,30,50]
V1_thres=[7,15,30]
V1_pools=[4,4,6]
V2_feature_maps=[15,40]
V2_thres=[7,15]
"""
2.多尺度深度脉冲特征提取算法
    a.使用3x3,5x5,7x7卷积，池化层都使用2x2，异步训练，最后contact
    b.用三种尺寸，每一种尺寸对应一组3X3,5X5的卷积核，最后contact
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

def train_v1(scale_id,train_loader):
    W1_name = 'v1_' + str(scale_id) + '_1.pkl'
    if not os.path.exists(os.path.join(weight_save_root_path, W1_name)):
        W1 = np.random.normal(0.8, 0.05, [V1_feature_maps[scale_id], 2, V1_kernel_sizes[scale_id], V1_kernel_sizes[scale_id]])
    else:
        W1 = read(os.path.join(weight_save_root_path, W1_name))

    m = model.measure(W1)
    if m >= C_l:
        max_epoches = 10000
        for epoch in range(1, max_epoches + 1):
            i = 0
            for (images, labels) in train_loader:
                input = images[0][0].numpy()

                i += 1
                temporal_coding = model.temporal_dog_coding(input,Tmax,spike_thres[scale_id])
                # print('编码层正确')

                temporal_coding = torch.Tensor(temporal_coding).permute(2,0,1)

                conv_time1, conv_V1 = model.spike_conv(temporal_coding,W1,V1_thres[scale_id],Tmax,dt)
                # print('卷积层正确')
                W1 = model.STDP(temporal_coding, conv_time1, conv_V1, W1)
                # print('正确训练')
                m = model.measure(W1)
                print('V1 并行'+str(scale_id)+'层,epoch ' + str(epoch) + ', 第', str(i), '次训练loss = ', str(m))
                if m < 0:
                    print('学习率过大，请调小')
                    exit(0)
                if epoch > 1:
                    if m < C_l:
                        # 说明收敛
                        print('V1 并行'+str(scale_id)+'层,epoch ' + str(epoch) + ', 第', str(i), '次训练  '+'收敛，停止运行')
                        write_pkl(os.path.join(weight_save_root_path, W1_name), W1)
                        return

def train_v2(scale_id,train_loader):
    W1_name = 'v2_' + str(scale_id) + '_1.pkl'
    if not os.path.exists(os.path.join(weight_save_root_path, W1_name)):
        W1 = np.random.normal(0.8, 0.05, [V2_feature_maps[0], 2, 3, 3])
    else:
        W1 = read(os.path.join(weight_save_root_path, W1_name))
    m = model.measure(W1)
    if m >= C_l:
        max_epoches = 10000
        flag = 0
        for epoch in range(1, max_epoches + 1):
            i = 0
            for (images, labels) in train_loader:
                input = images[0][0].numpy()
                i += 1
                input = cv2.resize(input, (0, 0), fx=1 - spike_thres[scale_id], fy=1 - spike_thres[scale_id],
                                   interpolation=cv2.INTER_NEAREST)
                temporal_coding = model.temporal_dog_coding(input,Tmax,0)
                # print('编码层正确')

                temporal_coding = torch.Tensor(temporal_coding).permute(2,0,1)
                conv_time1, conv_V1 = model.spike_conv(temporal_coding,W1,V2_thres[0],Tmax,dt)

                # print('卷积层正确')
                W1 = model.STDP(temporal_coding, conv_time1, conv_V1, W1)
                # print('正确训练')
                m = model.measure(W1)
                print('V2 并行'+str(scale_id)+'层 No.1 ,epoch ' + str(epoch) + ', 第', str(i), '次训练loss = ', str(m))
                if m < 0:
                    print('学习率过大，请调小')
                    exit(0)
                if epoch > 1:
                    if m < C_l:
                        # 说明收敛
                        print('V2 并行'+str(scale_id)+'层 No.1,epoch ' + str(epoch) + ', 第', str(i), '次训练  '+'收敛，停止运行')
                        write_pkl(os.path.join(weight_save_root_path, W1_name), W1)
                        flag = 1
                        break
            if flag:
                break

    W2_name = 'v2_' + str(scale_id) + '_2.pkl'
    if not os.path.exists(os.path.join(weight_save_root_path, W2_name)):
        W2 = np.random.normal(0.8, 0.05,[V2_feature_maps[1], V2_feature_maps[0], 5, 5])
    else:
        W2 = read(os.path.join(weight_save_root_path, W2_name))

    m = model.measure(W2)
    if m >= C_l:
        max_epoches = 10000
        for epoch in range(1, max_epoches + 1):
            i = 0
            for (images, labels) in train_loader:
                input = images[0][0].numpy()

                i += 1
                input = cv2.resize(input, (0, 0), fx=1-spike_thres[scale_id], fy=1-spike_thres[scale_id], interpolation=cv2.INTER_NEAREST)
                temporal_coding = model.temporal_dog_coding(input, Tmax, 0)
                # print('编码层正确')

                temporal_coding = torch.Tensor(temporal_coding).permute(2, 0, 1)

                conv_time1, conv_V1 = model.spike_conv(temporal_coding, W1, V2_thres[0], Tmax, dt)
                # print('卷积层正确')
                pool_time1 = model.spike_pooling(conv_time1,2,2)
                conv_time2, conv_V2 = model.spike_conv(pool_time1, W2, V2_thres[1], Tmax, dt)
                #pool_time2 = model.spike_pooling(conv_time2, 2, 2)
                #print(pool_time2.shape)
                W2 = model.STDP(pool_time1, conv_time2, conv_V2, W2)
                # print('正确训练')
                m = model.measure(W2)
                print('V2 并行' + str(scale_id) + '层 No.2,epoch ' + str(epoch) + ', 第', str(i), '次训练loss = ', str(m))
                if m < 0:
                    print('学习率过大，请调小')
                    exit(0)
                if epoch > 1:
                    if m < C_l:
                        # 说明收敛
                        print('V2 并行' + str(scale_id) + '层 No.2,epoch ' + str(epoch) + ', 第', str(i), '次训练  ' + '收敛，停止运行')
                        write_pkl(os.path.join(weight_save_root_path, W2_name), W2)
                        return

# 同尺寸，同层不同尺寸卷积核
class SCNN_Extractor_V1(object):
    def __init__(self):
        self.train_monitor = None

    def train(self,data_loader):
        for id in range(len(spike_thres)):
            train_v1(id, data_loader)
    def extractor(self,train_loader,test_loader):
        W1 = read(os.path.join(weight_save_root_path, 'v1_0_1.pkl'))
        W2 = read(os.path.join(weight_save_root_path, 'v1_1_1.pkl'))
        W3 = read(os.path.join(weight_save_root_path, 'v1_2_1.pkl'))
        W = [W1,W2,W3]
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i,(images , labels) in enumerate(train_loader):
            print('train',i)
            feature_tmp = []
            for scale_id in range(len(spike_thres)):
                input = images[0][0].numpy()
                temporal_coding = model.temporal_dog_coding(input, Tmax, spike_thres[scale_id])
                temporal_coding = torch.Tensor(temporal_coding).permute(2, 0, 1)
                conv_time1, conv_V1 = model.spike_conv(temporal_coding, W[scale_id], V1_thres[scale_id], Tmax, dt)
                pool_time1 = model.spike_pooling(conv_time1,V1_pools[scale_id],V1_pools[scale_id])
                scale_feauture= pool_time1.reshape(-1)
                feature_tmp.append(scale_feauture)
            train_x.append(np.concatenate(feature_tmp).tolist())
            train_y.append(labels[0])

        for i,(images , labels) in enumerate(test_loader):
            print('test',i)
            feature_tmp = []
            for scale_id in range(len(spike_thres)):
                input = images[0][0].numpy()
                temporal_coding = model.temporal_dog_coding(input, Tmax, spike_thres[scale_id])
                temporal_coding = torch.Tensor(temporal_coding).permute(2, 0, 1)
                conv_time1, conv_V1 = model.spike_conv(temporal_coding, W[scale_id], V1_thres[scale_id], Tmax, dt)
                pool_time1 = model.spike_pooling(conv_time1, V1_pools[scale_id], V1_pools[scale_id])
                scale_feauture = pool_time1.reshape(-1)
                feature_tmp.append(scale_feauture)
            test_x.append(np.concatenate(feature_tmp).tolist())
            test_y.append(labels[0])

        print(len(train_x))
        print(len(test_x))
        write_pkl('mnist_classify_train_data.pkl',[train_x,train_y])
        write_pkl('mnist_classify_test_data.pkl',[test_x,test_y])

    # def forward(self):

# 不同尺寸，同层同尺寸卷积核
class SCNN_Extractor_V2(object):
    def __init__(self):
        self.train_monitor = None

    def train(self,data_loader):
        for id in range(len(spike_thres)):
            train_v2(id, data_loader)
    def extractor(self,train_loader,test_loader):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i,(images , labels) in enumerate(train_loader):
            print('train',i)
            feature_tmp = []
            for scale_id in range(len(spike_thres)):
                W1 = read(os.path.join(weight_save_root_path, 'v2_'+str(scale_id)+'_1.pkl'))
                W2 = read(os.path.join(weight_save_root_path, 'v2_'+str(scale_id)+'_2.pkl'))
                input = images[0][0].numpy()
                input = cv2.resize(input, (0, 0), fx=1 - spike_thres[scale_id], fy=1 - spike_thres[scale_id],
                                   interpolation=cv2.INTER_NEAREST)
                temporal_coding = model.temporal_dog_coding(input, Tmax, 0)
                temporal_coding = torch.Tensor(temporal_coding).permute(2, 0, 1)
                conv_time1, conv_V1 = model.spike_conv(temporal_coding, W1, V2_thres[0], Tmax, dt)
                pool_time1 = model.spike_pooling(conv_time1,2,2)
                conv_time2, conv_V2 = model.spike_conv(pool_time1, W2, V2_thres[1], Tmax, dt)
                pool_time2 = model.spike_pooling(conv_time2, 2, 2)
                scale_feauture= pool_time2.reshape(-1)
                feature_tmp.append(scale_feauture)
            train_x.append(np.concatenate(feature_tmp).tolist())
            train_y.append(labels[0])

        for i,(images , labels) in enumerate(test_loader):
            print('test',i)
            feature_tmp = []
            for scale_id in range(len(spike_thres)):
                W1 = read(os.path.join(weight_save_root_path, 'v2_'+str(scale_id)+'_1.pkl'))
                W2 = read(os.path.join(weight_save_root_path, 'v2_'+str(scale_id)+'_2.pkl'))
                input = images[0][0].numpy()
                input = cv2.resize(input, (0, 0), fx=1 - spike_thres[scale_id], fy=1 - spike_thres[scale_id],
                                   interpolation=cv2.INTER_NEAREST)
                temporal_coding = model.temporal_dog_coding(input, Tmax, 0)
                temporal_coding = torch.Tensor(temporal_coding).permute(2, 0, 1)
                conv_time1, conv_V1 = model.spike_conv(temporal_coding, W1, V2_thres[0], Tmax, dt)
                pool_time1 = model.spike_pooling(conv_time1,2,2)
                conv_time2, conv_V2 = model.spike_conv(pool_time1, W2, V2_thres[1], Tmax, dt)
                pool_time2 = model.spike_pooling(conv_time2, 2, 2)
                scale_feauture= pool_time2.reshape(-1)
                feature_tmp.append(scale_feauture)
            test_x.append(np.concatenate(feature_tmp).tolist())
            test_y.append(labels[0])

        print(len(train_x))
        print(len(test_x))
        print(len(train_x[0]))
        write_pkl('v2_train_data.pkl',[train_x,train_y])
        write_pkl('v2_test_data.pkl',[test_x,test_y])

    # def forward(self):
