import math

import torch.nn as nn
import torch
from dotmap import DotMap

from eutils.container import AADModel
from eutils.snn.stbp.functional import TdBatchNorm1d, ModuleLayer, LILinearCell
from eutils.snn.stbp.lsnn_stbp import LSNNRecurrent, LSNNParameters
from eutils.snn.stbp.neurons import LIF


class Model(nn.Module):
    def __init__(self, local: DotMap):
        super(Model, self).__init__()
        self.local = local
        self.batch_size = None
        self.need_sa = local.model_meta.need_sa
        self.lstm_hidden = 10
        self.multiband = local.data_meta.eeg_band > 1
        from eutils.snn.stbp.setting import SNNParameters
        self.sp = SNNParameters(
            snn_process=local.model_meta.snn_process,
            vth_dynamic=False,
            vth_init=local.model_meta.vth,
            tau_mem=local.model_meta.tau_mem,
            tau_syn=local.model_meta.tau_syn,
            vth_low=0.1,
            vth_high=0.9,
            vth_random_init=False
        )
        self.attn_hidden = self.lstm_hidden
        self.input_size = local.data_meta.eeg_band_chan
        self.step = int(local.split_meta.time_len * local.data_meta.eeg_fs)
        self.output_size = 2
        self.left = 4
        self.right = 4

        # todo 测一下各种参数的影响 beta为0更好
        self.lstm = LSNNRecurrent(self.lstm_hidden, self.lstm_hidden,
                                  p=LSNNParameters(v_th=torch.as_tensor(self.sp.vth_init), beta=torch.as_tensor(0)), sp=self.sp)
        self.embedding_front = nn.Linear(self.input_size, self.lstm_hidden)
        # self.embedding_front = nn.Conv1d(self.input_size, self.input_size, kernel_size=17, groups=self.input_size)
        self.embedding_back = nn.Linear(self.input_size, self.lstm_hidden)
        self.classify = nn.Linear(self.lstm_hidden, self.output_size)

        # self_attention
        self.embedding_query = nn.Linear(self.lstm_hidden, self.attn_hidden, bias=False)
        self.embedding_key = nn.Linear(self.lstm_hidden, self.attn_hidden, bias=False)
        self.embedding_value = nn.Linear(self.lstm_hidden, self.lstm_hidden, bias=False)
        self.decoder_cma = nn.Linear(self.lstm_hidden, self.lstm_hidden, bias=False)

        # BN
        self.BNi = TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNq = TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNk = TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNv = TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNc = TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNo = TdBatchNorm1d(self.output_size, sp=self.sp)

        # snn
        self.embedding_front_s = ModuleLayer(self.embedding_front, sp=self.sp, bn=self.BNi)
        self.embedding_back_s = ModuleLayer(self.embedding_back, sp=self.sp)
        # self.classify_s = slayers.ModuleLayer(self.classify, sp=self.sp, bn=self.BN2)
        self.classify_s = LILinearCell(self.lstm_hidden * local.data_meta.eeg_band, self.output_size, sp=self.sp)
        self.embedding_query_s = ModuleLayer(self.embedding_query, sp=self.sp, bn=self.BNq)
        self.embedding_key_s = ModuleLayer(self.embedding_key, sp=self.sp, bn=self.BNk)
        self.embedding_value_s = ModuleLayer(self.embedding_value, sp=self.sp, bn=self.BNv)
        self.decoder_cma_s = ModuleLayer(self.decoder_cma, sp=self.sp, bn=self.BNc)

        self.spike_tanh1 = LIF(activation=nn.Tanh(), sp=self.sp)
        self.spike_tanh2 = LIF(activation=nn.Tanh(), sp=self.sp)
        self.spike_elu = LIF(activation=nn.ELU(), sp=self.sp)
        self.spike_softmax = LIF(activation=nn.Softmax(dim=1), sp=self.sp)
        self.spike_sigmoid = LIF(activation=nn.Sigmoid(), sp=self.sp)
        self.spike_relu1 = LIF(activation=nn.ReLU(), sp=self.sp)
        self.spike_relu2 = LIF(activation=nn.ReLU(), sp=self.sp)
        self.spike_relu3 = LIF(activation=nn.ReLU(), sp=self.sp)
        self.spike_relu4 = LIF(activation=nn.ReLU(), sp=self.sp)
        self.spike_gelu = LIF(activation=nn.GELU(), sp=self.sp)

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     for m in self.modules():
    #         if m is not None and isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight.data, 1)
    #             nn.init.zeros_(m.bias.data)

    # x shape: (Batch, Channel, Time)
    def update_qkv(self, x):
        q = self.embedding_query_s(x)
        if self.sp.snn_process:
            q = self.spike_relu1(q)
        k = self.embedding_key_s(x)
        if self.sp.snn_process:
            k = self.spike_relu2(k)
        v = self.embedding_value_s(x)
        if self.sp.snn_process:
            v = self.spike_tanh2(v)

        return q, k, v

    def dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 去掉这个softmax：有效
        attn_weights = torch.matmul(q.transpose(-1, -2), k)
        attn_weights = attn_weights / self.lstm_hidden
        # attn_weights = self.spike_softmax(attn_weights)

        # 去掉+x：效果不明显
        attn_value = torch.matmul(attn_weights, v.transpose(-1, -2))
        attn_value = attn_value.transpose(-1, -2)

        # 仿照lif神经元
        if self.sp.snn_process:
            attn_value = self.spike_relu3(attn_value)

        return attn_value, attn_weights

    def truncated_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        qf = q.permute(0, 2, 1).unsqueeze(2)
        conv = nn.Unfold(kernel_size=(self.lstm_hidden, self.left + 1 + self.right), padding=(0, self.left))
        kf = conv(k.unsqueeze(1))
        kf = kf.view(self.batch_size, self.lstm_hidden, self.left + 1 + self.right, kf.shape[-1])
        kf = kf.permute(0, 3, 1, 2)
        vf = conv(v.unsqueeze(1))
        vf = vf.view(self.batch_size, self.lstm_hidden, self.left + 1 + self.right, vf.shape[-1])
        vf = vf.permute(0, 3, 2, 1)

        attn_weights = torch.matmul(qf, kf)
        attn_weights = attn_weights / self.lstm_hidden
        attn_value = torch.matmul(attn_weights, vf)
        attn_value = attn_value.squeeze().transpose(1, 2)

        # 仿照lif神经元
        attn_value = self.spike_relu3(attn_value)

        return attn_value, attn_weights

    # x shape: (Batch, Channel, Time)
    def self_attention(self, x):
        q, k, v = self.update_qkv(x)
        attn_value, attn_weights = self.dot_product(q, k, v)
        attn_value = self.decoder_cma_s(attn_value)
        # if self.sp.snn_process:
        attn_value = self.spike_relu4(attn_value)
        return attn_value, attn_weights

    def truncated_self_attention(self, x):
        q, k, v = self.update_qkv(x)
        attn_value, attn_weights = self.truncated_dot_product(q, k, v)
        # step = q.shape[-1]
        # attn_value = []
        # attn_weights = []
        # for i in range(step):
        #     qq = q[:, :, i: i + 1]
        #     start = i - self.left if self.left is not None else 0
        #     end = i + 1 + self.right if self.right is not None else q.shape[-1]
        #     kk = k[:, :, max(0, start): min(step, end)]
        #     vv = v[:, :, max(0, start): min(step, end)]
        #     value, weight = self.dot_product(qq, kk, vv)
        #     attn_value.append(value)
        #     pad = ((start < 0) * abs(start), (end >= step) * abs(end - step))
        #     attn_weights.append(nn.functional.pad(weight, pad))
        # attn_value = torch.concat(attn_value, dim=-1)
        # print((attn_value == av).all())
        # # attn_weights = torch.concat(attn_weights, dim=1)
        attn_value = self.decoder_cma_s(attn_value)
        attn_value = self.spike_relu4(attn_value)
        return attn_value, attn_weights


    def forward(self, bs1, bs2, beeg, targets):
        visualization_weights = []
        # visualization_weights.append(DotMap(data=data[0].transpose(1, 2).clone(), title="A", need_save=True,
        #                                     figsize=[data[0].shape[1] * 2, data[0].shape[2] * 2]))
        self.batch_size = beeg.shape[0]
        eeg = beeg

        # 多频带分离
        if self.multiband:
            eeg = eeg.view(self.batch_size, self.local.data_meta.eeg_band, self.local.data_meta.eeg_band_chan, self.step)
        eeg_index = [i for i in range(len(eeg.shape))]

        # 这里可能需要一个编码器 v1lif性能明显下降 约60% v2bsa 失败
        # eeg, origin = self.bsa_encoder.encode(eeg)

        # wav and eeg shape: (Batch, Channel, Time)
        # embedding
        # embedding可以更复杂点：略微下降，效果不明显
        eeg = self.embedding_front_s(eeg)
        # eeg = self.spike_gelu(eeg)
        # eeg = self.embedding_back_s(eeg)
        eeg = self.spike_tanh1(eeg)

        # eeg shape: (Batch, Channel, Time)
        # SA
        if self.need_sa:
            eeg, weight = self.self_attention(eeg)

        # eeg shape: (Batch, Channel, Time)
        # lstm
        eeg = eeg.permute(eeg_index[-1:] + eeg_index[:-1]).contiguous()
        lstm_output, new_state = self.lstm(eeg)
        lstm_output = lstm_output.permute(eeg_index[1:] + eeg_index[:1]).contiguous()

        # eeg shape: (Batch, Channel, Time)
        output = lstm_output
        # output = self.classify_s(output)
        # output = self.spike_sigmoid(output)

        if self.multiband:
            output = output.view(self.batch_size, -1, self.step)

        if self.sp.snn_process:
            so = None
            temp = []
            for ts in range(output.shape[-1]):
                vo, so = self.classify_s(output[:, :, ts], so)
                temp.append(vo)
            output = torch.stack(temp, dim=2)
        else:
            output = self.classify(output.transpose(-1, -2)).transpose(-1, -2)
            output = self.spike_relu2(output)

        # output shape: (Batch, Channel, Time)
        # 反编码
        # todo 有没有其他反编码方法
        output = torch.sum(output, dim=2) / self.step
        # output, _ = torch.max(output, 2)

        # 性能明显下降 约68%
        # output = self.decoder(output).squeeze()

        return output, targets, visualization_weights


def get_model(local: DotMap) -> AADModel:
    model = Model(local)
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=torch.optim.Adam(model.parameters(), lr=local.lr),
        sched=None,
        dev=torch.device('cpu')
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model
