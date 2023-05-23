import numpy as np
import torch
import cv2

alpha=0.2
beta =0.8
def new_latency(map,Tmax):
    # print(data)
    # exit(0)

    map[map<0]=0
    coding=(1-np.sin(map*np.pi/2))*Tmax
    return coding


class STDP_extractor(object):
    def __init__(self,a_plus=0.004,a_minus=-0.003):
        self.a_plus = a_plus
        self.a_minus = a_minus

    def temporal_dog_coding(self,image,Tmax,pixl_thre):
        # 输入数据范围0-255

        if len(np.shape(image))==3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[image<pixl_thre]=0
        k1 = cv2.GaussianBlur(image, (7, 7), 1)
        k2 = cv2.GaussianBlur(image, (7, 7), 2)
        # 计算出 on-center map 和 off-center map
        on_center_map = cv2.subtract(k1, k2)
        off_center_map = cv2.subtract(k2, k1)
        # new latency coding

        dog1 = new_latency(on_center_map, Tmax)
        dog2 = new_latency(off_center_map, Tmax)
        for i in range(len(dog1)):
            for j in range(len(dog1[0])):
                if dog1[i][j] > dog2[i][j]:
                    dog1[i][j] = Tmax
                else:
                    dog2[i][j] = Tmax
        temporal_input_data = np.stack([dog1, dog2], 2)  # 获得了编码后的时间输入数据
        return temporal_input_data

    # @autojit
    def spike_conv(self,input,W,threshold,tmax,dt):
        # 脉冲卷积层
        kernel_size = W.shape[-1]
        input = torch.Tensor(input).unsqueeze(0)
        W = torch.Tensor(W)
        # print(W.shape[0],input.shape[2]-kernel_size+1,input.shape[3]-kernel_size+1)
        conv_time = np.ones([W.shape[0],input.shape[2]-kernel_size+1,input.shape[3]-kernel_size+1])*tmax
        conv_V = np.zeros(shape=conv_time.shape)
        total_step = int(tmax/dt)
        for step in range(total_step+1):
            # 该处可以理解为将原公式展开得到
            t= step*dt
            # 先计算膜电压
            S = t-input # S<=0,取值0; S>0,qu
            S = torch.where(S <= 0, torch.full_like(S,0),torch.full_like(S, 1)) # 充分考虑 k<t
            voltage = torch.conv2d(input=S,weight=W,stride=1)
            if torch.max(voltage)>=threshold:
                voltage = voltage.squeeze().numpy()
                v_tmp=np.reshape(voltage,[-1])
                conv_time_tmp = np.reshape(conv_time,[-1])
                conv_V_tmp = np.reshape(conv_V,[-1])
                fire_index = np.argwhere(v_tmp >= threshold).reshape([-1])
                current_index = np.argwhere(conv_time_tmp<t).reshape([-1])
                current_index = np.setdiff1d(fire_index,current_index)
                conv_time_tmp[current_index]=t
                conv_V_tmp[current_index]=v_tmp[current_index]
                conv_time = np.reshape(conv_time_tmp,voltage.shape)
                conv_V = np.reshape(conv_V_tmp,voltage.shape)
        # 竞争关系
        for i in range(conv_time.shape[1]):
            for j in range(conv_time.shape[2]):
                min_time = np.min(conv_time[:,i,j])
                if not (min_time==tmax):
                    min_index = np.argwhere(conv_time[:,i,j]==min_time).reshape([-1])
                    if len(min_index)>1:              #最小点火时间不止一个
                        # min_time_V = np.max(conv_V[:,i,j])
                        # k = np.argwhere(conv_V[:,i,j]==min_time_V).reshape([-1])
                        min_time_V = np.max(conv_V[min_index, i, j]) # 取其中最大电压值
                        k = min_index[np.argwhere(conv_V[min_index, i, j] == min_time_V).reshape([-1])[0]]
                    else:
                        k = min_index[0]
                    conv_time[:,i,j] = tmax
                    conv_time[k,i,j] = min_time
        return conv_time, conv_V

    def spike_pooling(self,input ,size,stride):
        # 脉冲池化层
        output = -torch.max_pool2d(input=torch.from_numpy(-input), kernel_size=size, stride=stride, ceil_mode=True)
        return output.numpy()

    def STDP(self, input_spike_time, output_spike_time, output_conv_V, W):
        input_spike_time = torch.Tensor(input_spike_time).numpy()
        feature_map_size = np.shape(output_spike_time)
        tmax = np.max(output_spike_time)
        input_channel = np.shape(W)[1]  # 输入的数据通道数
        output_channel = np.shape(W)[0]  # 输出的通道数/卷积层的卷积核个数
        kernel_step = np.shape(W)[2]
        W = np.array(W)
        input = np.array(input_spike_time)
        for i in range(output_channel):
            map_time = output_spike_time[i,:,:]
            min_time = np.min(map_time)
            if not(min_time==tmax):
                map_time_tmp = np.reshape(map_time,[-1])
                min_time_index = np.argwhere(map_time_tmp == min_time).reshape([-1])
                if len(min_time_index)==1:
                    index = min_time_index[0]
                else:
                    map_V = output_conv_V[i,:,:]
                    map_V_tmp = np.reshape(map_V, [-1])
                    max_vol = np.max(map_V_tmp[min_time_index])
                    index = min_time_index[np.argwhere(map_V_tmp[min_time_index] == max_vol).reshape([-1])[0]]
                #将index 转换为二维坐标
                irow = index//feature_map_size[-1]
                icol = index%feature_map_size[-1]
                j_time = input[:,irow:irow+kernel_step, icol:icol+kernel_step]
                mod_weight = self.single_pair_STDP(min_time, j_time, kernel_step, W[i,:,:,:])
                W[i, :, :, :] += mod_weight

        return W

    # @autojit
    def single_pair_STDP(self,post_time, pre_time,n,weight ):
        # a_plus = 0.004# 0.004
        # a_minus = -0.003#-0.003
        time_matrix = post_time - pre_time
        delta_W = np.where(time_matrix>=0,self.a_plus * weight*(1-weight),self.a_minus * weight*(1-weight))
        return delta_W

    def global_spike_conv(self,input, W, tmax):
        # 全局脉冲卷积层
        W =torch.Tensor(W)
        input =torch.Tensor(input).unsqueeze(0)
        # 该处可以理解为将原公式展开得到
        # 先计算膜电压
        S = tmax - input
        S = torch.where(S <= 0, torch.full_like(S,0), torch.full_like(S,1))  # 充分考虑 k<t
        voltage = torch.conv2d(input=S,weight=W,stride=1)
        return voltage.squeeze().numpy()

    def global_spike_pooling(self,input):
        # 全局脉冲池化层,最后池化为1x1xn的数据
        input = np.array(input)
        deepth = np.shape(input)[0]  # 输出的通道数/卷积层的卷积核个数
        global_pool = np.zeros([deepth])
        for i in range(deepth):
            global_pool[i]=np.max(input[i,:,:])
        return global_pool

    # @autojit
    def measure(self,W):
        """收敛性度量"""
        # 一个四维数组
        W= np.array(W)
        return np.average(np.multiply(W,1-W))
