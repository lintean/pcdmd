"""
3.新型基于脉冲的新型分类算法
    a.基于单脉冲的bp算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# batch 版本
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device =torch.device('cuda:1')
# device =torch.device('cpu')
loss_thres=0.5#0.07
# 常调试参数
batch_size = 32 # 60
learning_rate = 1e-3# Adam :5e-3
num_epochs = 50  # max epoch
sigma = 0.2 #0.14##0.1  # 0.1 0.05
# sigmas =[0.04,0.06,0.08,0.1,0.12,0.14,0.16]
#  不常调试参数
thresh = 1  # neuronal threshold
num_classes = 10
dt = .02
Tmax = 3 #6
encodemax = 3
time_window = int(Tmax / dt)
encode_window = 3
nonspike = Tmax
TimeLine = torch.range(0, Tmax / dt, 1) * dt
TimeLine = TimeLine.to(device)
magnitude = 1
num_timeline = TimeLine.shape[0]
tensorZero = torch.tensor(0.).to(device)
tensorOne = torch.tensor(1.).to(device)
Timeline_reverse = (TimeLine.flip(0) + 0.1).to(device)
# drop_rate =0.2

cfg_fc = [1876, 1024, 10]

interval = 0.2
"""

img = traindata(:,:,i)';
        img = 255-img;
        img(img<=80) = 255;
        img = tmax*img/255;
        img(img==tmax)=Inf;
        TrainDataCode(:,i) = img(:);

"""

class Dropout_Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, drop_prob=0, training=False):

        if training:
            mask = torch.rand(size=x.shape).gt(drop_prob).float().to(device)  # 0 1矩阵
            x = x + 1  # 去掉 0
            x = x * mask
            x = x - 1
            x[x < 0] = Tmax
            out = x
            ctx.save_for_backward(mask,)

            return out
        else:
            mask = torch.ones_like(x).float().to(device)
            ctx.save_for_backward(mask,)
            return x


    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors

        return grad_output*mask,None,None

dropout = Dropout_Spike.apply

def mask_kernel(channel, height, width):
    index_kernel = torch.arange(channel * height * width * 3).reshape(channel * height * width, 3)

    index_kernel[:, 0] = index_kernel[:, 0] / (height * width * 3) % channel
    index_kernel[:, 1] = index_kernel[:, 1] / (width * 3) % height
    index_kernel[:, 2] = index_kernel[:, 2] / 3 % width

    full_indices = torch.cat((index_kernel, index_kernel), 1).t().numpy().tolist()
    kernel_mask = torch.zeros((channel, height, width, channel, height, width))
    kernel_mask[full_indices] = 1

    return kernel_mask.reshape((channel * height * width, channel, height, width)).to(device)



def voltage_relay(voltage_con, weight):
    voltage_transpose = voltage_con.permute(0, 2, 1)
    weight_transpose = weight.t()
    voltage_post = voltage_transpose.matmul(weight_transpose)
    voltage_post = voltage_post.permute(0, 2, 1)
    return voltage_post

def seek_spike(voltage_post):
    voltage_binary = torch.where(voltage_post >= thresh, tensorOne , tensorZero )
    voltage_binary[:, :, -1] = 1.
    voltage_binary *= Timeline_reverse
    spike_time = voltage_binary.argmax(2).type(torch.float)
    return spike_time.to(device) * dt

def seek_spike_conv(mem):

    voltage_binary = torch.where(mem >= thresh, tensorOne , tensorZero )
    voltage_binary[:, :, :, :, -1] = 1.
    voltage_binary *= Timeline_reverse
    spike_time = voltage_binary.argmax(4).type(torch.float)

    return spike_time.to(device) * dt

class Linear_Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ti, W):  # dropout 用于全连接层
            # 计算膜电压

            shape = ti.shape
            # ti = ti.view(batch_size, -1)
            subtract_input = torch.repeat_interleave(ti.reshape(shape[0], shape[1], 1), num_timeline, dim=2)
            tmp = F.relu(TimeLine - subtract_input)
            mem = voltage_relay(tmp, W)
            # 计算脉冲to
            out = seek_spike(mem)
            ctx.save_for_backward(torch.autograd.Variable(out),torch.autograd.Variable(ti),torch.autograd.Variable(W), )
            return out

    @staticmethod
    def backward(ctx, grad_output):
        # 注意该处的grad_output 为dE/dto
        out, ti, W, = ctx.saved_tensors
        dEdto = grad_output
        full_out = torch.repeat_interleave(out.reshape(out.shape[0], out.shape[1], 1), ti.shape[1], dim=2)
        full_ti = torch.repeat_interleave(ti.reshape(ti.shape[0], 1, ti.shape[1]), out.shape[1], dim=1)

        dvdw = F.relu(full_out - full_ti)
        mask = dvdw.gt(0).float()

        # #该函数中需要反传的是dt/dv
        dtdv = -1 / (torch.mul(mask, W).sum(dim=2))
        dtdv = torch.where(out == nonspike, tensorZero, dtdv)
        dtdv = dtdv.clamp(min=-1)

        # 这一步需要反向计算的梯度是 dv/dti
        # b*o*i
        dvdti = mask * (-torch.repeat_interleave(W.reshape(1, W.shape[0], W.shape[1]),ti.shape[0], dim=0))

        dvdti = torch.where(torch.isnan(dvdti) == 1, torch.zeros_like(dvdti), dvdti)

        if dEdto.shape[1] == cfg_fc[-1]:
            ind_label = dEdto > 0
            ind_nonspike = out == nonspike
            ind_target = ind_label.float() * ind_nonspike.float()
            dtdv = torch.where(ind_target == 1, -tensorOne , dtdv)

        dvdw = torch.where(dvdw == nonspike, torch.zeros_like(dvdw), dvdw)

        dvdw = dvdw.clamp(max=2)
        dvdw = torch.where(torch.isnan(dvdw) == 1, torch.zeros_like(dvdw), dvdw)

        delta = dEdto * dtdv
        dE = delta.reshape(delta.shape[0], 1, delta.shape[1]).matmul(dvdti)
        dE = dE.squeeze(dim=1)

        deriv = torch.mul(torch.repeat_interleave(delta.reshape(delta.shape[0], delta.shape[1], 1), dvdw.shape[2], dim=2), dvdw)
        deriv = torch.sum(deriv, dim=0)
        #deriv /= ti.shape[0] #batch_size

        return dE, deriv

class Conv2d_Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ti, W, stride=1):  # dropout 用于全连接层
        # 计算膜电压
        shape = ti.shape
        tmp = F.relu(TimeLine - ti.reshape(shape[0], shape[1], shape[2], shape[3], 1))

        # size in b*time*channels*width*height
        tmp = tmp.permute(0, 4, 1, 2, 3)
        ts = tmp.shape
        tmp = tmp.reshape(ts[0] * ts[1], ts[2], ts[3], ts[4])
        mem = F.conv2d(tmp, W,stride=stride)
        mem = mem.reshape(shape[0], num_timeline, mem.shape[1], mem.shape[2], mem.shape[3]).permute(0, 2, 3, 4, 1)
        # out_shape = mem.shape

        out = seek_spike_conv(mem)

        ctx.save_for_backward(torch.autograd.Variable(out), torch.autograd.Variable(ti), torch.autograd.Variable(W), torch.autograd.Variable(torch.tensor(stride)))

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 注意该处的grad_output 为dE/dto
        out, ti, W, stride = ctx.saved_tensors
        # 求出dt/dv、dv/dw、dv/dti
        W_shape = W.shape
        shape = ti.shape
        stride = int(stride)
        conv_window = W_shape[2]
        out_channel = W_shape[0]
        in_channel = W_shape[1]
        out_shape = out.shape
        # 求出dv/dw
        input_full = F.conv2d(ti, mask_kernel(in_channel, conv_window, conv_window),stride=stride) \
            .reshape(shape[0], 1, shape[1], conv_window, conv_window, out_shape[2], out_shape[3])

        # size in b*o*i*5*5*h'*w'

        dvdw = F.relu(out.reshape(out_shape[0], out_shape[1], 1, 1, 1, out_shape[2], out_shape[3]) - input_full)
        # size in b*o*i*h'*w'*5*5
        mask = dvdw.permute(0, 1, 2, 5, 6, 3, 4).gt(0).float()
        # 求出dt/dv

        dtdv = torch.mul(mask, W.reshape(1, W.shape[0], W.shape[1], 1, 1, conv_window, conv_window))

        dtdv = dtdv.sum(dim=6).sum(dim=5).sum(dim=2)
        # size in b*o*h'*w'
        dtdv = -1 / dtdv
        dtdv = torch.where(out == nonspike, tensorZero, dtdv)
        dtdv = dtdv.clamp(min=-1)

        # 求出dv/dti
        step = math.ceil(((shape[-1]-1)*stride+conv_window-out_shape[-1])/2)#conv_window - 1
        # print('out.shape : ',out_shape)
        # print('step : ',step)
        out_padding = torch.nn.functional.pad(out, (step,step, step, step),'constant', 0)
        # print('out_padding.shape : ',out_padding.shape)
        # print('mask.shape : ',mask_kernel(out_channel, conv_window, conv_window).shape)
        mask_out2in = F.conv2d(out_padding, mask_kernel(out_channel, conv_window, conv_window),stride=stride)
        # print('mask_in : ',mask_out2in.shape)
        # print('reshape : ',batch_size, out_shape[1], 1, conv_window,conv_window, shape[2], shape[3])
        # exit(0)
        mask_out2in=mask_out2in.reshape( out_shape[0], out_shape[1], 1, conv_window,conv_window, shape[2], shape[3])
        mask_out2in = (mask_out2in - ti.reshape(shape[0], 1, shape[1], 1, 1, shape[2], shape[3])).gt(0).float()
        dvdti = mask_out2in.permute(0, 1, 2, 5, 6, 3, 4) * (
            torch.flip(-W, [2, 3]).reshape(1, W_shape[0], W_shape[1], 1, 1, conv_window, conv_window))

        dvdti = torch.where(torch.isnan(dvdti) == 1, tensorZero, dvdti)

        conv_window = W_shape[2]
        out_channel = W_shape[0]
        in_channel = W_shape[1]
        # b*o*h'*w'

        dEdto = grad_output

        dvdw = torch.where(dvdw == nonspike, tensorZero, dvdw)

        dvdw = dvdw.clamp(max=2)
        dvdw = torch.where(torch.isnan(dvdw) == 1, tensorZero, dvdw)

        # why does this variable contain nan?
        # size in b*o*h'*w'
        delta = dEdto * dtdv

        dvdw_shape = dvdw.shape
        deriv = torch.mul(delta.reshape(dvdw_shape[0], dvdw_shape[1], 1, 1, 1, dvdw_shape[5], dvdw_shape[6]), dvdw)
        deriv = deriv.sum(dim=6).sum(dim=5).sum(dim=0)

        delta_padding = torch.nn.functional.pad(delta, (step, step, step,step), 'constant', 0)
        vti_shape = dvdti.shape

        delta_padding = F.conv2d(delta_padding, mask_kernel(out_channel, conv_window, conv_window),stride=stride). \
            reshape(vti_shape[0], vti_shape[1], 1, conv_window, conv_window, vti_shape[3], vti_shape[4])
        dE = delta_padding.permute(0, 1, 2, 5, 6, 3, 4) * dvdti
        dE = dE.sum(dim=6).sum(dim=5).sum(dim=1)

        deriv *= (magnitude / vti_shape[0])#batch_size

        return dE, deriv,None

Linear = Linear_Spike.apply
Conv2d = Conv2d_Spike.apply



# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=40):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


# fc layer


# cfg_fc = [2, 3, 1]

class SpikeMLP(nn.Module):
    def __init__(self):
        super(SpikeMLP, self).__init__()
        self.fc1_weight = nn.Parameter(torch.tensor(np.random.normal(0.01*0.2, 0.01*0.2, [cfg_fc[1], cfg_fc[0]])).float(),
                                       requires_grad=True)
        self.fc2_weight = nn.Parameter(torch.tensor(np.random.normal(0.01*0.1, 0.01*0.1, [cfg_fc[2], cfg_fc[1]])).float(),
                                       requires_grad=True)
        # 1.
        # self.fc1_weight = nn.Parameter(torch.tensor(np.random.normal(0, 2/(cfg_fc[1]+cfg_fc[0]), [cfg_fc[1], cfg_fc[0]])).float(),
        #                                requires_grad=True)
        # self.fc2_weight = nn.Parameter(torch.tensor(np.random.normal(0, 2/(cfg_fc[2]+cfg_fc[1]), [cfg_fc[2], cfg_fc[1]])).float(),
        #                                requires_grad=True)
        # 2.
        # self.fc1_weight = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, [cfg_fc[1], cfg_fc[0]])).float(),
        #                                requires_grad=True)
        # self.fc2_weight = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, [cfg_fc[2], cfg_fc[1]])).float(),
        #                                requires_grad=True)
        # 3
        # self.fc1_weight = nn.Parameter(torch.tensor(np.random.normal(0, 0.001, [cfg_fc[1], cfg_fc[0]])).float(),
        #                                requires_grad=True)
        # self.fc2_weight = nn.Parameter(torch.tensor(np.random.normal(0, 0.001, [cfg_fc[2], cfg_fc[1]])).float(),
        #                                requires_grad=True)
        # self.fc1_weight = nn.Parameter(torch.rand(cfg_fc[1], cfg_fc[0]) * 0.7 - 0.2, requires_grad=True)
        # self.fc2_weight = nn.Parameter(torch.rand(cfg_fc[2], cfg_fc[1]) * 0.7- 0.2, requires_grad=True)
        self.linear_spike = Linear

    def forward(self, input):
        # current_batch = input.shape[0]
        # ti = latency_coding(input).view(current_batch,-1)
        ti = input
        h1_spike = self.linear_spike(ti, self.fc1_weight)
        # h1_spike = dropout(h1_spike, drop_rate, self.training)
        h2_spike = self.linear_spike(h1_spike, self.fc2_weight)
        #print('h1',h1_spike)
        #print('h2',h2_spike)
        #exit(0)
        outputs = h2_spike
        return outputs


class loss_fun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, label):
        # 计算出loss
        to = output.detach()
        mask = label.detach()
        a = F.softmax(-to, dim=1)

        loss_distri = -torch.log((mask * a).sum(dim=1))

        loss = loss_distri.sum() / batch_size

        is_update = (loss_distri < loss_thres)

        # 反传dE/dto
        a = torch.where(to == nonspike, 0 * torch.ones_like(to), a)
        ctx.save_for_backward(a, mask, is_update)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        a, mask, is_update = ctx.saved_tensors
        dEdt = torch.where(mask == 1, 1 - a, -a)
        ud_mask = torch.repeat_interleave(is_update.reshape(is_update.shape[0], 1), dEdt.shape[1], dim=1)
        dEdt[ud_mask] = 0
        return dEdt, None,


