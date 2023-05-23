##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import torch.nn as nn
from snn import LinearBN1d,  ConvBN2d_AvgPool, sDropout
from functional import ZeroExpandInput_CNN
from dotmap import DotMap


class sCNN(nn.Module):
    def __init__(self, num_labels, T):
        super(sCNN, self).__init__()
        self.T = T
        self.out_num = num_labels
        self.encoder1 = ConvBN2d_AvgPool(1, 32, (3, 3), pooling=2)# or ConvBN2d_MaxPool
        self.encoder2 = ConvBN2d_AvgPool(16, 32, (3, 3), pooling=2)
        self.sDropoutLinear = sDropout(2, pDrop=0.30)
        self.fc1 = LinearBN1d(15*15*32, 512)
        self.fc2 = LinearBN1d(512, 32)
        self.output = nn.Linear(32, num_labels)

    def forward(self, x):
        visualization_weights = []
        # visualization_weights.append(
        #     DotMap(data=x.clone().cpu().detach().numpy(), title="origin", need_save=True))

        # 扩展出多一个维度T，应该是snn所用的脉冲时间
        x_spike, x = ZeroExpandInput_CNN.apply(x, self.T)
        x_spike, x = self.encoder1(x_spike, x)
        # x_spike, x = self.encoder2(x_spike, x)
        # visualization_weights.append(DotMap(data=x_spike.clone().cpu().detach().numpy(), title="conv", kernel_size=3*3, c_in=1, c_out=32, need_save=True, attach_result=True))
        x_spike, x = self.sDropoutLinear(x_spike, x)
        x_spike = x_spike.view(x_spike.size(0), self.T, -1)
        x = x.view(x.size(0), -1)
        x_spike, x = self.fc1(x_spike, x)
        # visualization_weights.append(DotMap(data=x_spike.clone().cpu().detach().numpy(), title="fc1", c_in=7200, c_out=512))
        x_spike, x = self.fc2(x_spike, x)
        # visualization_weights.append(DotMap(data=x_spike.clone().cpu().detach().numpy(), title="fc2", c_in=512, c_out=32))
        # visualization_weights.append(DotMap(title="fc3", c_in=32, c_out=self.out_num))
        return self.output(x), visualization_weights





