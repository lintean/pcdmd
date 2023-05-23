import math
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch
from dotmap import DotMap
from eutils.torch.container import AADModel
from eutils.util import get_gpu_with_max_memory


# 整体模型
from modules.transformer import TransformerEncoder


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.conv_output_channel = 10
        self.conv_eeg_audio_number = 4

        self.ofc_channel = args.window_length - args.eeg_conv_size // 2 * 2

        self.output_fc = nn.Sequential(
            nn.Linear(self.ofc_channel * 2, self.ofc_channel), nn.ReLU(),
            nn.Linear(self.ofc_channel, 2), nn.Sigmoid()
        )

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        mult = math.ceil(args.window_length / 128)
        self.conv1 = nn.Conv1d(1, 8, 40, stride=20)
        self.conv2 = nn.Conv1d(8, args.input_size, 18, stride=3)

        self.proj_images = nn.Conv1d(args.eeg_channel, args.input_size, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(args.eeg_channel, args.input_size, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(args.audio_channel, args.input_size, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(args.audio_channel, args.input_size, 1, bias=False)

        self.audio_conv = nn.Conv1d(args.input_size, args.cls_channel, args.audio_conv_size)
        self.eeg_conv = nn.Conv1d(args.input_size, args.cls_channel, args.eeg_conv_size)

        audio_fs = args.audio_fs if "audio_fs" in args else 8000
        self.wav_conv = nn.Unfold(kernel_size=(1, audio_fs), stride=(1, audio_fs))

    def dot(self, a, b):
        # 皮尔逊相似度
        # mean_a = torch.mean(a, dim=0)
        # mean_b = torch.mean(b, dim=0)
        # a = a - mean_a
        # b = b - mean_b
        # 余弦
        cos = torch.matmul(a.transpose(-1, -2), b)
        cos = torch.diagonal(cos, dim1=-1, dim2=-2)
        norm_a = torch.norm(a, p=2, dim=-2)
        norm_b = torch.norm(b, p=2, dim=-2)
        cos = cos / (norm_a * norm_b)
        return cos

    def Euclidean_Distance(self, a, b):
        # 欧氏距离
        temp = a - b
        temp = torch.norm(temp, p=2, dim=0)
        return temp

    def wav_encoder(self, wav):
        wav = self.wav_conv(wav[:, None, ...])[:, None, ...]
        wav_temp = []
        for i in range(wav.shape[-1]):
            wav_temp.append(self.conv2(self.conv1(wav[..., i])))
        return torch.concat(wav_temp, dim=-1)

    def forward(self, bs1, bs2, beeg, targets):
        visualization_weights = []
        wavA = bs1
        eeg = beeg
        wavB = bs2

        # wav and eeg shape: (Batch, Channel, Time), wav Channel 1 to 16
        if "encoder" in self.args and self.args.encoder:
            wavA = self.wav_encoder(wavA)
            wavB = self.wav_encoder(wavB)
        else:
            wavA = self.proj_audio(wavA)
            wavB = self.proj_audio(wavB)

        eeg = self.proj_images(eeg)

        eeg = self.eeg_conv(eeg)
        wavA = self.audio_conv(wavA)
        wavB = self.audio_conv(wavB)
        data = [wavA, eeg, eeg, wavB]

        # dot
        # wav and eeg shape: (Batch, Channel, Time)
        data_dot = None
        for i in range(2):
            temp1 = self.dot(data[i * 3], data[i + 1])
            temp1 = self.fc[i](temp1.unsqueeze(-1))
            data_dot = temp1.squeeze(-1) if data_dot is None else torch.cat([data_dot, temp1.squeeze(-1)], dim=-1)
        output = self.output_fc(data_dot)

        return output, targets, visualization_weights


def get_model(args: DotMap) -> AADModel:
    model = Model(args)
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=torch.optim.Adam(model.parameters(), lr=args.lr),
        sched=None,
        dev=torch.device(get_gpu_with_max_memory(args.gpu_list))
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model
