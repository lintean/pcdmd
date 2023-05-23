import torch.nn as nn
import math
import eutils.util as util
import eutils.split_utils as sutil
from dotmap import DotMap
import torch.fft
import time
import torch
from eutils.lsnn_stbp import LSNNRecurrent
from eutils.lsnn_stbp import LSNNParameters
import eutils.snn_layers_stbp as slayers
from eutils.snn_layers_stbp import LILinearCell

# from norse.torch.module.lsnn import LSNNRecurrent
# from norse.torch.functional.lsnn import LSNNParameters
# from norse.torch.module.leaky_integrator import LILinearCell

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_document_path = device_to_use.origin_data_document + "/DTU_single_single_snn_1to32"
# args.data_document_path = device_to_use.origin_data_document + "/KUL_single_single_snn_1to32_mean"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "LSTM_v3"
args.ConType = ["No"]

args.CNN_file = "./CNN_normal.py"
args.CNN_split_file = "./CNN_split.py"
args.data_document = device_to_use.splited_data_document + "/????"

# 加载数据集元数据
data_meta = util.read_json(args.data_document_path + "/metadata.json")
args.data_meta = data_meta
args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
# args.names = ["S14"]
args.need_sa = True

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
args.window_length = math.ceil(data_meta.fs * 2)
args.window_lap = math.ceil(data_meta.fs * 0.2)
# args.window_lap = None
args.overlap = 1 - args.window_lap / args.window_length if args.window_lap is not None else 0
args.delay = 0
args.batch_size = 32
args.max_epoch = [100]
args.lr = [1e-3]
args.early_patience = 0
args.random_seed = time.time()
args.cross_validation_fold = 5
# args.current_flod = 0
args.one_hot_target = False

# 可视化选项 列表为空表示不希望可视化
args.visualization_epoch = []
args.visualization_window_index = []

# 非常用参数，分别是 被试数量、通道数量、trail数量、trail内数据点数量、测试集比例、验证集比例
# 一般不需要调整
args.people_number = data_meta.people_number
args.eeg_band = data_meta.eeg_band
args.eeg_channel_per_band = data_meta.eeg_channel_per_band
args.eeg_channel = args.eeg_band * args.eeg_channel_per_band
args.audio_band = data_meta.audio_band
args.audio_channel_per_band = data_meta.audio_channel_per_band
args.audio_channel = args.audio_band * args.audio_channel_per_band
args.channel_number = args.eeg_channel + args.audio_channel * 2
args.trail_number = data_meta.trail_number
args.cell_number = data_meta.cell_number
args.bands_number = data_meta.bands_number
args.fs = data_meta.fs
args.test_percent = 0.2
args.vali_percent = 0

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
args.isDS = True
# DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
args.isFM = 0 if "KUL" in args.data_document_path else 1
print(args.isFM)
# 回归模型还是分类模型
args.normalization = False

# 数据划分选择
# 测试集划分是否跨trail
args.isBeyoudTrail = False
# 是否使用100%的数据作为训练集，isBeyoudTrail=False、isALLTrain=True、need_pretrain = True、need_train = False说明跨被试
args.isALLTrain = False

# 预训练选择
# 只有train就是单独训练、只有pretrain是跨被试、两者都有是预训练
# 跨被试还需要上方的 isALLTrain 为 True
args.need_pretrain = False
args.need_train = True

# 预处理步骤
args.process_steps = [sutil.get_data_from_preprocess, sutil.subject_split]

# 尝试启用gpu
# device = util.select_device(device_to_use.gpu_list)

# device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
device = torch.device('cpu')

def add_constraint(model):
    pass
    # low, high = model.sp.vth_low, model.sp.vth_high
    # constraints = util.WeightConstraint(low=low, high=high)
    #
    # model.spike_tanh1.apply(constraints)
    # model.spike_tanh2.apply(constraints)
    # model.spike_elu.apply(constraints)
    # model.spike_softmax.apply(constraints)
    # model.spike_sigmoid.apply(constraints)
    # model.spike_relu1.apply(constraints)
    # model.spike_relu2.apply(constraints)
    # model.spike_relu3.apply(constraints)
    # model.spike_relu4.apply(constraints)
    # model.spike_gelu.apply(constraints)
    # model.lstm.apply(constraints)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.need_sa = args.need_sa
        self.lstm_hidden = 10
        self.multiband = args.eeg_band > 1
        self.sp = slayers.SNNParameters(
            snn_process=True,
            vth_dynamic=False,
            vth_init=0.5,
            vth_low=0.1,
            vth_high=0.9,
            vth_random_init=False
        )
        self.attn_hidden = self.lstm_hidden
        self.input_size = args.eeg_channel_per_band
        self.step = args.window_length
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
        self.decoder = nn.Linear(self.step, 1)

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
        q = self.spike_relu1(q)
        k = self.embedding_key_s(x)
        k = self.spike_relu2(k)
        v = self.embedding_value_s(x)
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
        attn_value = self.spike_relu3(attn_value)

        return attn_value, attn_weights

    def truncated_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        qf = q.permute(0, 2, 1).unsqueeze(2)
        conv = nn.Unfold(kernel_size=(self.lstm_hidden, self.left + 1 + self.right), padding=(0, self.left))
        kf = conv(k.unsqueeze(1))
        kf = kf.view(args.batch_size, self.lstm_hidden, self.left + 1 + self.right, kf.shape[-1])
        kf = kf.permute(0, 3, 1, 2)
        vf = conv(v.unsqueeze(1))
        vf = vf.view(args.batch_size, self.lstm_hidden, self.left + 1 + self.right, vf.shape[-1])
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

        eeg = beeg[:, 0, :, :]

        # 多频带分离
        if self.multiband:
            eeg = eeg.view(args.batch_size, args.eeg_band, args.eeg_channel_per_band, self.step)
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
        eeg = eeg.permute(eeg_index[-1:] + eeg_index[:-1])
        lstm_output, new_state = self.lstm(eeg)
        lstm_output = lstm_output.permute(eeg_index[1:] + eeg_index[:1])

        # eeg shape: (Batch, Channel, Time)
        output = lstm_output
        # output = self.classify_s(output)
        # output = self.spike_sigmoid(output)

        if self.multiband:
            output = output.view(args.batch_size, -1, self.step)

        so = None
        temp = []
        for ts in range(output.shape[-1]):
            vo, so = self.classify_s(output[:, :, ts], so)
            temp.append(vo)
        output = torch.stack(temp, dim=2)

        # output shape: (Batch, Channel, Time)
        # 反编码
        # todo 有没有其他反编码方法
        output = torch.sum(output, dim=2) / args.window_length
        # output, _ = torch.max(output, 2)

        # 性能明显下降 约68%
        # output = self.decoder(output).squeeze()

        return output, targets, visualization_weights


# 模型参数和初始化
myNet = [CNN()]
clip = 0.8

lr = args.lr
optimzer = [torch.optim.Adam(myNet[i].parameters(), lr=lr[i]) for i in range(len(myNet))]

# scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimzer[0], gamma=0.999), torch.optim.lr_scheduler.ExponentialLR(optimzer[1], gamma=0.999)]
scheduler = [None]
loss_func = [nn.CrossEntropyLoss()]
# torch.autograd.set_detect_anomaly(True)
