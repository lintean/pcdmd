import torch
import torch.nn as nn
import math
import eutils.util as util
from dotmap import DotMap
import torch.fft
import time
import eutils.snn_layers as slayers
import eutils.snn_lstm as slstm

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_document_path = device_to_use.origin_data_document + "/KUL_LSTM_TEST"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "LSTM_test_v2"
args.ConType = ["No"]

args.CNN_file = "./CNN_normal.py"
args.CNN_split_file = "./CNN_split.py"
args.data_document = device_to_use.splited_data_document + "/????"

# 加载数据集元数据
data_meta = util.read_json(args.data_document_path + "/metadata.json")
args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
# args.names = ["S10"]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
args.window_length = math.ceil(data_meta.fs * 1)
# args.window_lap = math.ceil(data_meta.fs * 0.5)
args.window_lap = None
args.overlap = 1 - args.window_lap / args.window_length if args.window_lap is not None else 0
args.delay = 0
args.batch_size = 32
args.max_epoch = 100
args.min_epoch = 0
args.early_patience = 0
args.random_seed = time.time()
args.cross_validation_fold = 1
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
args.vali_percent = 0.2

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
args.isDS = True
# DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
args.isFM = 0
# 回归模型还是分类模型
args.is_regression = False

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

# 启用gpu
device = torch.device('cuda:' + str(util.get_gpu_with_max_memory(device_to_use.gpu_list)))
# device = torch.device('cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.need_sa = False
        self.snn_process = True
        self.lstm_hidden = 10
        self.attn_hidden = self.lstm_hidden
        self.step_length = 8
        self.output_size = 2
        self.vth = 0.2

        self.embedding = nn.Linear(args.eeg_channel, self.lstm_hidden)
        self.time_avg = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classify = nn.Linear(self.lstm_hidden, self.output_size)
        self.do = nn.Dropout(1)

        # self_attention
        self.embedding_query = nn.Linear(self.lstm_hidden, self.attn_hidden, bias=False)
        self.embedding_key = nn.Linear(self.lstm_hidden, self.attn_hidden, bias=False)
        self.embedding_value = nn.Linear(self.lstm_hidden, self.lstm_hidden, bias=False)

        # snn
        self.bn0 = slayers.TdBatchNorm1d(self.lstm_hidden, snn_process=self.snn_process)
        self.bn1 = slayers.TdBatchNorm1d(self.output_size, snn_process=self.snn_process)
        self.embedding_s = slayers.ModuleLayer(self.embedding, snn_process=self.snn_process)
        self.lstm_s1 = slstm.SLSTM(self.vth, self.snn_process, self.lstm_hidden, self.lstm_hidden, 1)
        self.lstm_s2 = slstm.SLSTM(self.vth, self.snn_process, self.lstm_hidden, self.lstm_hidden, 1)
        self.time_avg_s = slayers.ModuleLayer(self.time_avg, snn_process=self.snn_process)
        self.flatten_s = slayers.ModuleLayer(self.flatten, snn_process=self.snn_process)
        self.classify_s = slayers.ModuleLayer(self.classify, snn_process=self.snn_process)
        self.embedding_query_s = slayers.ModuleLayer(self.embedding_query, snn_process=self.snn_process)
        self.embedding_key_s = slayers.ModuleLayer(self.embedding_key, snn_process=self.snn_process)
        self.embedding_value_s = slayers.ModuleLayer(self.embedding_value, snn_process=self.snn_process)

        self.spike_tanh = slayers.LIFSpike(snn_process=self.snn_process, activation=nn.Tanh(), vth=self.vth)
        self.spike_elu = slayers.LIFSpike(snn_process=self.snn_process, activation=nn.ELU(), vth=self.vth)
        self.spike_softmax = slayers.LIFSpike(snn_process=self.snn_process, activation=nn.Softmax(dim=1), vth=self.vth)
        self.spike_sigmoid = slayers.LIFSpike(snn_process=self.snn_process, activation=nn.Sigmoid(), vth=self.vth)

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     for m in self.modules():
    #         if m is not None and isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight.data, 1)
    #             nn.init.zeros_(m.bias.data)

    def self_attention(self, x):
        q = self.embedding_query_s(x)
        q = self.spike_elu(q)
        k = self.embedding_key_s(x)
        k = self.spike_elu(k)
        v = self.embedding_value_s(x)
        v = self.spike_tanh(v)

        attn_weights = slayers.opt_layer(torch.bmm, q, k.transpose(1, 2), snn_process=self.snn_process)
        attn_weights = attn_weights / (args.eeg_channel ** 0.5)
        attn_weights = self.spike_softmax(attn_weights)
        # todo 这里差一个Linear
        # todo 去掉+x
        attn_value = slayers.opt_layer(torch.bmm, attn_weights, v, snn_process=self.snn_process) + x

        return attn_value, attn_weights

    def forward(self, x):
        visualization_weights = []
        # visualization_weights.append(DotMap(data=data[0].transpose(1, 2).clone(), title="A", need_save=True,
        #                                     figsize=[data[0].shape[1] * 2, data[0].shape[2] * 2]))

        wavA = x[:, 0, 0:args.audio_channel, :]
        wavA = wavA.transpose(1, 2)
        eeg = x[:, 0, args.audio_channel:-args.audio_channel, :]
        eeg = eeg.transpose(1, 2)
        wavB = x[:, 0, -args.audio_channel:, :]
        wavB = wavB.transpose(1, 2)

        # wav and eeg shape: (Batch, Time, Channel)
        # todo 这里要参考论文
        # SNN编码转换
        if self.snn_process:
            eeg = torch.chunk(eeg, self.step_length, dim=1)
            eeg = torch.stack(eeg, dim=3)

        # eeg shape: (Batch, T, Channel, Step)

        # embedding
        eeg = self.embedding_s(eeg)
        # eeg = self.bn0(eeg.transpose(1, 2)).transpose(1, 2)
        eeg = self.spike_tanh(eeg)

        # eeg shape: (Batch, T, self.lstm_length, Step)
        # SA
        if self.need_sa:
            eeg, weight = self.self_attention(eeg)

        # eeg shape: (Batch, T, self.lstm_length, Step)
        # lstm * 2
        eeg, _ = self.lstm_s1(eeg)
        # eeg, _ = self.lstm_s2(eeg)

        # eeg shape: (Batch, T, self.lstm_length, Step)
        # decoding and classification
        eeg = self.time_avg_s(eeg.transpose(1, 2))

        # eeg shape: (Batch, Channel = self.lstm_length, 1, Step)
        eeg = self.flatten_s(eeg)

        # eeg shape: (Batch, self.lstm_length, Step)
        # todo 参考dropout的论文
        output = eeg
        output = self.classify_s(output)
        # output = self.bn1(output)
        output = self.spike_sigmoid(output)

        # output shape: (Batch, self.output_size, Step)
        # 反编码
        if self.snn_process:
            output = torch.sum(output, dim=2) / self.step_length
        return output, visualization_weights


# 模型参数和初始化
myNet = CNN()
clip = 0.8

lr = 1e-3
optimzer = torch.optim.Adam(myNet.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()