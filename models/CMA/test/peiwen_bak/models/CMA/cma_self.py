import math
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch
from dotmap import DotMap
from scipy import signal

from eutils.hongxu.Audio_TCN_encoder import Audio_Encoder
from eutils.torch.container import AADModel
from eutils.util import get_gpu_with_max_memory, normalization
from pytorch_msssim import ssim


# 整体模型
from modules.transformer import TransformerEncoder


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim ** -0.5

        # cross_attention
        self.embedding_query = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.embedding_key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.embedding_value = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.decoder_cma = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def update_qkv(self, query, key, value):
        q = self.embedding_query(query)
        k = self.embedding_key(key)
        v = self.embedding_value(value)
        return q, k, v

    def dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q *= self.scaling
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        attn_value = torch.matmul(attn_weights, v)
        attn_value = attn_value.transpose(0, 1)

        return attn_value, attn_weights

    # x shape: Time x Batch x Channel
    def forward(self, query, key, value, **kwargs):
        q, k, v = self.update_qkv(query, key, value)
        attn_value, attn_weights = self.dot_product(q, k, v)
        attn_value = self.decoder_cma(attn_value)
        # attn_value = F.relu(attn_value)
        return attn_value, attn_weights


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.conv_output_channel = 10
        self.conv_eeg_audio_number = 4
        self.h = args.h

        if not args.transformer:
            self.cm_attn = nn.ModuleList([CrossAttention(
                embed_dim=self.h
            ) for i in range(self.conv_eeg_audio_number)])
        else:
            self.cm_attn = nn.ModuleList([TransformerEncoder(
                embed_dim=self.h,
                num_heads=1,
                layers=args.cma_layer,
                attn_dropout=0,
                relu_dropout=0,
                res_dropout=0,
                embed_dropout=0,
                attn_mask=False
            ) for i in range(self.conv_eeg_audio_number)])
        self.cma_layer = args.cma_layer if isinstance(self.cm_attn[0], CrossAttention) else 1
        print(f"cma_layer: {self.cma_layer}")

        self.channel = [self.h, self.h, self.h, self.h]
        self.ofc_channel = args.window_length

        self.output_conv_fc = nn.Sequential(
            nn.Linear(args.window_length, args.window_length), nn.ReLU(),
            nn.Linear(args.window_length, 2), nn.Sigmoid()
        )

        self.output_sum_fc = nn.Sequential(
            # nn.Linear(2, 2), nn.ReLU(),
            nn.Linear(2, 2), nn.Sigmoid()
        )

        self.judge_fc = nn.Sequential(
            nn.Linear(args.window_length, args.window_length), nn.ReLU(),
            nn.Linear(args.window_length, 1), nn.Sigmoid()
        )

        self.cos_conv = nn.Sequential(
            nn.Conv1d(2, 1, 17, padding='same'),
            nn.ReLU()
        )

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        self.fc2 = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        mult = math.ceil(args.window_length / 128)
        self.conv1 = nn.Conv1d(1, 1, 40, stride=20)
        self.conv2 = nn.Conv1d(1, 1, 18, stride=3)
        self.conv3 = nn.Conv1d(1, self.h, 20, stride=10)

        self.proj_images = nn.Conv1d(args.eeg_channel, self.h, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(args.eeg_channel, self.h, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(args.audio_channel, self.h, 1, bias=False)
        self.encoder = nn.Conv1d(args.audio_channel, self.h, 17, bias=False, padding=8)

        self.lstm_wavA = nn.LSTM(self.h, self.h, 8)
        self.lstm_wavB = nn.LSTM(self.h, self.h, 8)
        self.lstm_eeg = nn.LSTM(self.h, self.h, 8)

        audio_fs = args.audio_fs if "audio_fs" in args else 8000
        self.wav_conv = nn.Unfold(kernel_size=(1, audio_fs), stride=(1, audio_fs))
        self.cos_fc = nn.Linear(args.window_length, 1)

        # 红旭师兄的encoder
        H, P, X = 17, 3, 1
        self.audio_encoder = Audio_Encoder(self.h, H, P, X)

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
        wav = wav.cpu().detach().numpy()
        wav = signal.resample(wav, 1290 * self.args.time_length, axis=-1)
        audio_fs = 1290
        wav = torch.from_numpy(wav).to(self.args.dev)
        self.wav_conv = nn.Unfold(kernel_size=(1, audio_fs), stride=(1, audio_fs))

        wav = self.wav_conv(wav[:, None, ...])[:, None, ...]
        # wav = wav.transpose(1, -1)
        wav_temp = []
        for i in range(wav.shape[-1]):
            # wav_temp.append(self.conv2(self.conv1(wav[..., i])))
            wav_temp.append(self.conv3(wav[..., i]))
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


        # 4CMA
        # multihead_attention Input shape: Time x Batch x Channel
        # wav and eeg shape: (Batch, Channel, Time)
        data = [wavA, eeg, eeg, wavB]
        kv = [eeg, wavA, wavB, eeg]
        hash = {0: 0, 1: 1, 2: 1, 3: 0}
        weight = [0 for i in range(self.conv_eeg_audio_number)]
        for l in range(self.cma_layer):
            for i in range(self.conv_eeg_audio_number):
                data[i] = data[i].permute(2, 0, 1)
                kv[i] = kv[i].permute(2, 0, 1)
                data[i], weight[i] = self.cm_attn[hash[i]](data[i], kv[i], kv[i], return_=True)
                data[i] = data[i].permute(1, 2, 0)
                kv[i] = kv[i].permute(1, 2, 0)

        # dot
        # wav and eeg shape: (Batch, Channel, Time)
        data_dot = []
        for i in range(2):
            # cos
            temp1 = self.dot(data[i * 3], data[i + 1])
            data_dot.append(temp1)
        data_dot = torch.stack(data_dot, dim=1)

        # output process

        # conv + fc (cannot work)
        # output = self.cos_conv(data_dot).squeeze()
        # output = self.output_conv_fc(output)

        # mean + fc (maybe work, 68% or 72%) 可能是训练问题
        if torch.isnan(data_dot).all() or torch.isinf(data_dot).all():
            print("error")
        # output = data_dot.mean(dim=-1)
        # output = self.output_sum_fc(output)

        # sum (work, 77%)
        output = data_dot.sum(dim=-1)

        # fc (maybe work, 50%)
        # output = self.judge_fc(data_dot).squeeze()

        return output, targets, visualization_weights


def get_model(args: DotMap) -> AADModel:
    model = Model(args)
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2 if "l2" in args else 0),
        sched=None,
        dev=torch.device(get_gpu_with_max_memory(args.gpu_list))
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model
