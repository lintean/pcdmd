from STDP_extractor import STDP_extractor
import numpy as np
import os
import _pickle as pkl
import torch
import cv2
import time
import threading
Tmax=1
dt=.02
C_l=0.0005
spike_thres = [0,0,0]
V1_kernel_sizes=[5,9,16]
V1_feature_maps=[4,10,20]
V1_thres=[10,30,60]
V1_pools=[2,2,5]


parallel_loss_record1=[]
parallel_loss_record2=[]
"""
2.多尺度深度脉冲特征提取算法
    a.使用3x3,5x5,7x7卷积，池化层都使用2x2，异步训练，最后contact
    b.用三种尺寸，每一种尺寸对应一组5X5的卷积核，最后contact
"""
model = STDP_extractor(0.004,-0.003)
weight_save_root_path=r'./prallelweight'

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

def train_layer(layer,train_loader):
    if layer == 0:
        record_loss = parallel_loss_record1
    else:
        record_loss = parallel_loss_record2
    W1_name = 'v1_'+str(layer)+'_1.pkl'
    if not os.path.exists (os.path.join (weight_save_root_path, W1_name)):
        W1 = np.random.normal (0.8, 0.05, [V1_feature_maps[layer], 2, V1_kernel_sizes[layer], V1_kernel_sizes[layer]])
    else:
        W1 = read (os.path.join (weight_save_root_path, W1_name))

    m = model.measure (W1)

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

                conv_time1, conv_V1 = model.spike_conv (temporal_coding, W1, V1_thres[layer], Tmax, dt)
                # print('卷积层正确')
                W1 = model.STDP (temporal_coding, conv_time1, conv_V1, W1)
                # print('正确训练')
                m = model.measure (W1)
                record_loss.append (m)
                print ('V1 并行'+str(layer)+' 层,epoch ' + str (epoch) + ', 第', str (i), '次训练loss = ', str (m))
                if m < 0:
                    print ('学习率过大，请调小')
                    exit (0)
                if epoch > 1:
                    if m < C_l:
                        # 说明收敛
                        print ('V1 并行'+str(layer)+'层,epoch ' + str (epoch) + ', 第', str (i), '次训练  ' + '收敛，停止运行')
                        write_pkl (os.path.join (weight_save_root_path, W1_name), W1)
                        if  layer == 0:
                            n_ = '1'
                        else:
                            n_ ='2'
                        write_pkl ('Caltech_parallel_loss_record'+n_+'.pkl', record_loss)
                        return
paral_time=[]
import queue as Q
def record(x):
    with open('caltech_parallel_record.txt','a+') as f:
        f.write(str(x)+' ')
        f.flush()
        f.close()
def get_run_time():
    with open('caltech_parallel_record.txt','r') as f:
        row=f.readline().strip().split(' ')
        f.close()
        row=list(map(float,row))
        return max(row[0],row[1])+max(row[2],row[3])
def initial_record():
    with open('caltech_parallel_record.txt','w') as f:
        f.flush()
        f.close()
def forward(layer,train_loader,W1):


    for i, (images, labels) in enumerate (train_loader):
        print(i)
        input = images[0][0].numpy ()

        temporal_coding = model.temporal_dog_coding (input, Tmax, spike_thres[0])
        # print('编码层正确')
        temporal_coding = torch.Tensor (temporal_coding).permute (2, 0, 1)
        conv_time1, conv_V1 = model.spike_conv (temporal_coding, W1, V1_thres[layer], Tmax, dt)
        pool_time1 = model.spike_pooling (conv_time1, V1_pools[layer], V1_pools[layer])

    # record(end-start)
class train_Thread (threading.Thread):
    def __init__(self, i,train_loader):
        threading.Thread.__init__(self)
        self.threadID = i
        self.train_loader =train_loader

    def run(self):
        train_layer (self.threadID, self.train_loader)

class forward_Thread (threading.Thread):
    def __init__(self, i,image,W):
        threading.Thread.__init__(self)
        self.threadID = i
        self.image =image
        self.W = W

    def run(self):
        forward (0, self.image, self.W)
from multiprocessing import Pool
# 同尺寸，同层不同尺寸卷积核
class SCNN_Extractor_V1(object):
    def __init__(self):
        self.train_monitor = None

    def train(self,train_loader):
        start = time.time()
        pool = Pool (2)
        for i in range (2):
            pool.apply_async (train_layer , (i,train_loader))
        pool.close()
        pool.join()
        # thread1 = train_Thread (0, train_loader)
        # thread2 = train_Thread (1, train_loader)
        #
        # # 开启新线程
        # thread1.start ()
        # thread2.start ()
        # thread1.join ()
        # thread2.join ()

        print('并行结构训练耗时：',time.time()-start,' s')

    def extractor(self,train_loader,test_loader):
        W1 = read (os.path.join (weight_save_root_path, 'v1_0_1.pkl'))
        W2 = read (os.path.join (weight_save_root_path, 'v1_1_1.pkl'))
        # start = time.time ()
        accu = 0
        print ('train')
        pool = Pool (2)
        start = time.time()
        pool.apply_async (forward, (0, train_loader, W1))
        pool.apply_async (forward, (1, train_loader, W2))
        pool.close ()
        pool.join ()
        accu +=(time.time()-start)
        print ('test')
        pool = Pool (2)
        start=time.time()
        pool.apply_async (forward, (0, test_loader, W1))
        pool.apply_async (forward, (1, test_loader, W2))
        pool.close ()
        pool.join ()
        # accu=get_run_time()
        accu +=(time.time()-start)
        print ('每条数据前馈时间：', (accu) / (len (train_loader) + len (test_loader)), 's')