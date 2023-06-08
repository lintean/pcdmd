import math
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch
from dotmap import DotMap
from scipy import signal

from eutils.torch.container import AADModel
from eutils.util import get_gpu_with_max_memory
import pytorch_warmup as warmup


# 整体模型
from models.CMAA.modules.transformer import TransformerEncoder


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.out_proj = nn.LSTM(16, 16, 8)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None, **kwargs):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights = attn_weights + attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class Model(nn.Module):
    def __init__(self, local):
        super(Model, self).__init__()
        self.local = local
        dm = local.data_meta
        mm = local.model_meta
        self.mm, self.dm = mm, dm
        self.window_length = int(local.split_meta.time_len * dm.eeg_fs)
        self.conv_output_channel = 10
        self.conv_eeg_audio_number = 4
        self.h = mm.h

        if not mm.transformer:
            self.cm_attn = nn.ModuleList([MultiheadAttention(
                embed_dim=self.h,
                num_heads=1,
                attn_dropout=0
            ) for i in range(self.conv_eeg_audio_number)])
        else:
            self.cm_attn = nn.ModuleList([TransformerEncoder(
                embed_dim=self.h,
                num_heads=1,
                layers=mm.cma_layer,
                attn_dropout=0,
                relu_dropout=0,
                res_dropout=0,
                embed_dropout=0,
                attn_mask=False
            ) for i in range(self.conv_eeg_audio_number)])
        self.cma_layer = mm.cma_layer if isinstance(self.cm_attn[0], MultiheadAttention) else 1
        print(f"cma_layer: {self.cma_layer}")

        self.channel = [self.h, self.h, self.h, self.h]
        self.ofc_channel = self.window_length

        self.output_fc = nn.Sequential(
            nn.Linear(self.window_length * 2, self.window_length), nn.ReLU(),
            nn.Linear(self.window_length, 2), nn.Sigmoid()
        )

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        self.fc2 = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        mult = math.ceil(self.window_length / 128)
        self.conv1 = nn.Conv1d(1, 1, 40, stride=20)
        self.conv2 = nn.Conv1d(1, 1, 18, stride=3)
        self.conv3 = nn.Conv1d(1, self.h, 20, stride=10)

        self.proj_images = nn.Conv1d(dm.eeg_chan, self.h, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(dm.eeg_chan, self.h, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(dm.wav_chan, self.h, 1, bias=False)
        self.encoder = nn.Conv1d(dm.wav_chan, self.h, 17, bias=False, padding=8)

        self.lstm_wavA = nn.LSTM(self.h, self.h, 8)
        self.lstm_wavB = nn.LSTM(self.h, self.h, 8)
        self.lstm_eeg = nn.LSTM(self.h, self.h, 8)

        audio_fs = dm.wav_fs
        self.wav_conv = nn.Unfold(kernel_size=(1, audio_fs), stride=(1, audio_fs))
        self.cos_fc = nn.Linear(self.window_length, 1)

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
        if "encoder" in self.mm and self.mm.encoder:
            wavA = self.wav_encoder(wavA)
            wavB = self.wav_encoder(wavB)
            # wavA = self.proj_audio(wavA)
            # wavB = self.proj_audio(wavB)
        else:
            wavA = self.proj_audio(wavA)
            wavB = self.proj_audio(wavB)
            # wavA = self.encoder(wavA)
            # wavB = self.encoder(wavB)

        eeg = self.proj_images(eeg)

        # # LSTM input shape: Time x Batch x Channel
        # wavA = wavA.permute(2, 0, 1)
        # wavB = wavB.permute(2, 0, 1)
        # eeg = eeg.permute(2, 0, 1)
        # wavA, _ = self.lstm_wavA(wavA)
        # wavB, _ = self.lstm_wavA(wavB)
        # eeg, _ = self.lstm_eeg(eeg)
        # wavA = wavA.permute(1, 2, 0)
        # wavB = wavB.permute(1, 2, 0)
        # eeg = eeg.permute(1, 2, 0)

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
        data_dot = None
        for i in range(2):
            # cos + sum
            temp1 = self.dot(data[i * 3], data[i + 1])
            temp1 = temp1.sum(dim=-1)
            # temp1 = self.cos_fc(temp1).squeeze()
            data_dot = temp1 if data_dot is None else torch.stack([data_dot, temp1], dim=-1)

            # cos + fc
            # temp1 = self.dot(data[i * 3], data[i + 1])
            # data_dot = temp1 if data_dot is None else torch.concat([data_dot, temp1], dim=-1)

            # ssim
            # temp1 = ssim(normalization(data[i * 3].unsqueeze(1)), normalization(data[i + 1].unsqueeze(1)), data_range=1,
            #              size_average=False)
            # data_dot = temp1 if data_dot is None else torch.stack([data_dot, temp1], dim=-1)

        output = data_dot
        # output = self.output_fc(data_dot)

        return output, targets, visualization_weights


def get_model(args: DotMap) -> AADModel:
    model = Model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2 if "l2" in args else 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=optimizer,
        sched=scheduler,
        warmup=warmup_scheduler,
        dev=torch.device(get_gpu_with_max_memory(args.gpu_list))
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model
