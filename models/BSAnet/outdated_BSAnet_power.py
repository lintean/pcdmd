import torch
import torch.nn as nn
from dotmap import DotMap

import eutils.snn.snn_energy as sutil
import eutils.snn_layers_stbp as slayers
from eutils.lsnn_stbp import LSNNParameters
from eutils.lsnn_stbp import LSNNRecurrent
from eutils.snn.container import SPower
from eutils.snn_layers_stbp import LILinearCell
from eutils.torch.container import AADModel


class Model(nn.Module):
    def __init__(self, args: DotMap):
        super(Model, self).__init__()
        self.args = args
        self.need_sa = args.need_sa
        self.lstm_hidden = 10
        self.multiband = args.eeg_band > 1
        self.sp = slayers.SNNParameters(
            snn_process=args.snn_process,
            vth_dynamic=False,
            vth_init=args.vth,
            tau_mem=args.tau_mem,
            tau_syn=args.tau_syn,
            vth_low=0.1,
            vth_high=0.9,
            vth_random_init=False
        )
        self.powerp = {
            "snn_process": args.snn_process,
            "verbose": False
        }
        self.attn_hidden = self.lstm_hidden
        self.input_size = args.eeg_channel_per_band
        self.step = args.window_length
        self.output_size = 2
        self.left = 4
        self.right = 4

        # todo 测一下各种参数的影响 beta为0更好
        self.lstm = LSNNRecurrent(self.lstm_hidden, self.lstm_hidden,
                                  p=LSNNParameters(v_th=torch.as_tensor(self.sp.vth_init), beta=torch.as_tensor(0)),
                                  sp=self.sp)
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
        self.BNi = slayers.TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNq = slayers.TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNk = slayers.TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNv = slayers.TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNc = slayers.TdBatchNorm1d(self.lstm_hidden, sp=self.sp)
        self.BNo = slayers.TdBatchNorm1d(self.output_size, sp=self.sp)

        # snn
        self.embedding_front_s = slayers.ModuleLayer(self.embedding_front, sp=self.sp, bn=self.BNi)
        self.embedding_back_s = slayers.ModuleLayer(self.embedding_back, sp=self.sp)
        # self.classify_s = slayers.ModuleLayer(self.classify, sp=self.sp, bn=self.BN2)
        self.classify_s = LILinearCell(self.lstm_hidden * args.eeg_band, self.output_size, sp=self.sp)
        self.embedding_query_s = slayers.ModuleLayer(self.embedding_query, sp=self.sp, bn=self.BNq)
        self.embedding_key_s = slayers.ModuleLayer(self.embedding_key, sp=self.sp, bn=self.BNk)
        self.embedding_value_s = slayers.ModuleLayer(self.embedding_value, sp=self.sp, bn=self.BNv)
        self.decoder_cma_s = slayers.ModuleLayer(self.decoder_cma, sp=self.sp, bn=self.BNc)

        self.spike_tanh1 = slayers.LIFSpike(activation=nn.Tanh(), sp=self.sp)
        self.spike_tanh2 = slayers.LIFSpike(activation=nn.Tanh(), sp=self.sp)
        self.spike_elu = slayers.LIFSpike(activation=nn.ELU(), sp=self.sp)
        self.spike_softmax = slayers.LIFSpike(activation=nn.Softmax(dim=1), sp=self.sp)
        self.spike_sigmoid = slayers.LIFSpike(activation=nn.Sigmoid(), sp=self.sp)
        self.spike_relu1 = slayers.LIFSpike(activation=nn.ReLU(), sp=self.sp)
        self.spike_relu2 = slayers.LIFSpike(activation=nn.ReLU(), sp=self.sp)
        self.spike_relu3 = slayers.LIFSpike(activation=nn.ReLU(), sp=self.sp)
        self.spike_relu4 = slayers.LIFSpike(activation=nn.ReLU(), sp=self.sp)
        self.spike_gelu = slayers.LIFSpike(activation=nn.GELU(), sp=self.sp)

        self.energys = SPower()

    # x shape: (Batch, Channel, Time)
    def update_qkv(self, x):
        data_in = x.clone()
        q = self.embedding_query_s(x)
        q = self.spike_relu1(q)
        self.energys += sutil.energy(
            **self.powerp,
            data_in=data_in,
            data_out=data_in,
            module=self.embedding_query_s
        )

        data_in = x.clone()
        k = self.embedding_key_s(x)
        k = self.spike_relu2(k)
        self.energys += sutil.energy(
            **self.powerp,
            data_in=data_in,
            data_out=data_in,
            module=self.embedding_key_s
        )

        data_in = x.clone()
        v = self.embedding_value_s(x)
        v = self.spike_tanh2(v)
        self.energys += sutil.energy(
            **self.powerp,
            data_in=data_in,
            data_out=data_in,
            module=self.embedding_value_s
        )

        return q, k, v

    def dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 去掉这个softmax：有效
        data_in = k.clone()
        attn_weights = torch.matmul(q.transpose(-1, -2), k)
        attn_weights = attn_weights / self.lstm_hidden
        # attn_weights = self.spike_softmax(attn_weights)

        if self.sp.snn_process:
            self.energys += sutil.get_snn_energy(
                ann_power=sutil.energy_ann_fc(
                    c_in=self.lstm_hidden,
                    c_out=self.step
                ) * self.args.batch_size * self.step,
                firing_rate=sutil.get_firing_rate(
                    data_in,
                    return_spower=True
                )
            )
        else:
            self.energys += sutil.energy_ann_fc(
                c_in=self.step,
                c_out=self.lstm_hidden
            ) * self.args.batch_size * self.step

        # 去掉+x：效果不明显
        data_in = v.clone()
        attn_value = torch.matmul(attn_weights, v.transpose(-1, -2))
        attn_value = attn_value.transpose(-1, -2)

        # 仿照lif神经元
        if self.sp.snn_process:
            attn_value = self.spike_relu3(attn_value)

        if self.sp.snn_process:
            self.energys += sutil.get_snn_energy(
                ann_power=sutil.energy_ann_fc(
                    c_in=self.step,
                    c_out=self.lstm_hidden
                ) * self.args.batch_size * self.step,
                firing_rate=sutil.get_firing_rate(
                    data_in,
                    return_spower=True
                )
            )
        else:
            self.energys += sutil.energy_ann_fc(
                c_in=self.step,
                c_out=self.lstm_hidden
            ) * self.args.batch_size * self.step

        return attn_value, attn_weights

    def truncated_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        qf = q.permute(0, 2, 1).unsqueeze(2)
        conv = nn.Unfold(kernel_size=(self.lstm_hidden, self.left + 1 + self.right), padding=(0, self.left))
        kf = conv(k.unsqueeze(1))
        kf = kf.view(self.args.batch_size, self.lstm_hidden, self.left + 1 + self.right, kf.shape[-1])
        kf = kf.permute(0, 3, 1, 2)
        vf = conv(v.unsqueeze(1))
        vf = vf.view(self.args.batch_size, self.lstm_hidden, self.left + 1 + self.right, vf.shape[-1])
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

        data_in = attn_value.clone()
        attn_value = self.decoder_cma_s(attn_value)
        attn_value = self.spike_relu4(attn_value)

        self.energys += sutil.energy(
            **self.powerp,
            data_in=data_in,
            data_out=data_in,
            module=self.decoder_cma_s
        )
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

        eeg = beeg
        self.energys = SPower()

        # 多频带分离
        if self.multiband:
            eeg = eeg.view(self.args.batch_size, self.args.eeg_band, self.args.eeg_channel_per_band, self.step)
        eeg_index = [i for i in range(len(eeg.shape))]

        # 这里可能需要一个编码器 v1lif性能明显下降 约60% v2bsa 失败
        # eeg, origin = self.bsa_encoder.encode(eeg)

        # wav and eeg shape: (Batch, Channel, Time)
        eeg = self.embedding_front_s(eeg)
        eeg = self.spike_tanh1(eeg)

        self.energys += sutil.energy_ann_fc(
            c_in=self.input_size,
            c_out=self.lstm_hidden
        ) * self.args.batch_size * self.step

        # eeg shape: (Batch, Channel, Time)
        # SA
        if self.need_sa:
            eeg, weight = self.self_attention(eeg)

        # eeg shape: (Batch, Channel, Time)
        # lstm
        eeg = eeg.permute(eeg_index[-1:] + eeg_index[:-1]).contiguous()
        data_in = eeg.clone()
        lstm_output, new_state = self.lstm(eeg)
        if self.sp.snn_process:
            self.energys += sutil.get_snn_energy(
                ann_power=sutil.energy_ann_fc(
                    c_in=self.lstm_hidden,
                    c_out=self.lstm_hidden
                ) * 2 * self.args.batch_size * self.step,
                firing_rate=sutil.get_firing_rate(
                    data_in,
                    return_spower=True
                )
            )
        else:
            self.energys += sutil.energy(
                **self.powerp,
                data_in=data_in,
                data_out=lstm_output,
                module=self.lstm
            )
        lstm_output = lstm_output.permute(eeg_index[1:] + eeg_index[:1]).contiguous()

        # eeg shape: (Batch, Channel, Time)
        output = lstm_output

        if self.multiband:
            output = output.view(self.args.batch_size, -1, self.step)

        data_in = output.clone()
        so = None
        temp = []
        for ts in range(output.shape[-1]):
            vo, so = self.classify_s(output[:, :, ts], so)
            temp.append(vo)
        output = torch.stack(temp, dim=2)

        if self.sp.snn_process:
            self.energys += sutil.get_snn_energy(
                ann_power=sutil.energy_ann_fc(
                    c_in=self.lstm_hidden,
                    c_out=self.output_size
                ) * self.args.batch_size * self.step,
                firing_rate=sutil.get_firing_rate(
                    data_in,
                    return_spower=True
                )
            )
        else:
            self.energys += sutil.energy_ann_fc(
                c_in=self.lstm_hidden,
                c_out=self.output_size
            ) * self.args.batch_size * self.step

        # output shape: (Batch, Channel, Time)
        output = torch.sum(output, dim=2) / self.step

        return output, targets, self.energys


def get_model(args: DotMap) -> AADModel:
    model = Model(args)
    aad_model = AADModel(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optim=torch.optim.Adam(model.parameters(), lr=args.lr),
        sched=None,
        dev=torch.device('cpu')
    )

    # scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
    # device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
    return aad_model
