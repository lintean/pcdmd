import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

EPS = 1e-8   # avoid the case of divide 0

class Audio_Encoder(nn.Module):
    """Estimation the audio feature from envelope with TCN network
    """
    """
    Args:
        the envelope data : [M,1,T], M is batch size, T is length of the sample       
    Returns:
        the extracted latent representation for audio-eeg fusion  [M,B,T]， B is the number of channel that same with the EEG part
    Others:
    H:  H: Number of channels in inter convolutional blocks
    P:  Kernel size in convolutional blocks
    X： Number of convolutional blocks in each repeat
    """
    def __init__(self, B, H , P , X):
        super(Audio_Encoder, self).__init__()
        # Hyper-parameter
        self.B, self.H , self.P, self.X = B, H, P, X
        # Components
        # [M,1,T] -> [M, B, T]
        self.bottleneck_conv1x1 = nn.Conv1d(1, B, 1, bias=False)   # transform the dimension from audio to eeg
        # [M,1,T] -> [M, B, T]
        #self.norm_0 = ChannelWiseLayerNorm(N)    #channel layernorm
        self.tcn_1 = tcn(B, H, P, X)
        self.tcn_2 = tcn(B, H, P, X)
        self.tcn_3 = tcn(B, H, P, X)

    def forward(self, audio_file):
        """
        Args:
            the envelope data: [M, 1, T], M is batch size, T is length of the samples
        Returns:
            the extracted latent representation for audio-eeg fusion
        """
        audio_expand = self.bottleneck_conv1x1(audio_file)   # (M, 1, T)   -->   (M, B, T)

        # THREE repeats TO FORM THE FULL TCN
        audio_output = self.tcn_1(audio_expand)     # (M, B, T) --> (M, B, T)
        audio_output = self.tcn_2(audio_output)     # (M, B, T) --> (M, B, T)
        audio_output = self.tcn_3(audio_output)     # (M, B, T) --> (M, B, T)

        #audio_output = self.dependency_net(audio_expand)     # (M, B, T)
        return audio_output      # or try the audio_expand * audio_output


class tcn(nn.Module):
    def __init__(self, B, H , P , X):
        super(tcn, self).__init__()
        blocks = []
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            blocks += [TemporalBlock(B, H, P, stride=1,
                                     padding=padding,
                                     dilation=dilation)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.net(x)
        return out



class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):

        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu











# class Chomp1d(nn.Module):
#     """To ensure the output length is the same as the input.
#     """
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         """
#         Args:
#             x: [M, H, Kpad]
#         Returns:
#             [M, H, K]
#         """
#         return x[:, :, :-self.chomp_size].contiguous()
#
# class ChannelwiseLayerNorm(nn.Module):
#     """Channel-wise Layer Normalization (cLN)"""
#     def __init__(self, channel_size):
#         super(ChannelwiseLayerNorm, self).__init__()
#         self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
#         self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.gamma.data.fill_(1)
#         self.beta.data.zero_()
#
#     def forward(self, y):
#         """
#         Args:
#             y: [M, N, K], M is batch size, N is channel size, K is length
#         Returns:
#             cLN_y: [M, N, K]
#         """
#         mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
#         var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
#         cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
#         return cLN_y
#
class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                 pointwise_conv)

    def forward(self, x):
        return self.net(x)





if __name__ == "__main__":
    #check the function of the network
    M, C, T = 2, 64, 80   # Input parameter    M: Batch size, C: EEG channel number T: length of the segment
    torch.manual_seed(123)

    # the network parameter of the audio encoder
    B, H, P, X = 16, 32, 3, 4   # use the parameters here as the default setting

    # Generate the input data
    Audio_file = torch.randint(3,(M, T))   # envelope
    Audio_file = torch.unsqueeze(Audio_file, 1)   # (M, T) --> (M, 1, T)
    Audio_file = Audio_file.float()

    #Generate the network parameter
    audio_encoder = Audio_Encoder(B, H, P, X)
    audio_feature = audio_encoder(Audio_file)

