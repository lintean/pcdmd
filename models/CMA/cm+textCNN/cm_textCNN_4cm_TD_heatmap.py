import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder
from ecfg import gpu_random

# 使用的参数
# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
label = "cm_textCNN_4cm_TD_heatmap"
ConType = ["No"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
#          "S18"]
# names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"]
names = ["S3"]

# 所用的数据目录路径
# data_document_path = "../dataset"
data_document_path = "../dataset_16"
# data_document_path = "../dataset_csp"
# data_document_path = "../dataset_fbcsp"
# data_document_path = "../data_all_split"
# data_document_path = "/document/data/eeg/dataset"
# data_document_path = "/document/data/eeg/dataset_csp"
# data_document_path = "/document/data/eeg/dataset_fbcsp"
# data_document_path = "/document/data/eeg/dataset_fbcsp10"
# data_document_path = "/document/data/eeg/data_all_split"

CNN_file = "./CNN_normal_heatmap_attend.py"
CNN_split_file = "./CNN_split.py"
data_document = "./data_new"

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
overlap = 0.5
window_length = 140
delay = 0
batch_size = 1
max_epoch = 200
min_epoch = 0
isEarlyStop = False

TimeDelaySta = 0
TimeDelayEnd = 230
ken_length = 5

# 非常用参数，分别是 被试数量、通道数量、trail数量、trail内数据点数量、测试集比例、验证集比例
# 一般不需要调整
people_number = 18
eeg_channel = 16
audio_channel = 1
channel_number = eeg_channel + audio_channel * 2
trail_number = 20
cell_number = 3500
test_percent = 0.1
vali_percent = 0.1

# print(TimeDelaySta )
# print(TimeDelayEnd )
TimeDelaySta = int(TimeDelaySta / 1000 * 70)
TimeDelayEnd = int(TimeDelayEnd / 1000 * 70)
TimeDelayNum = TimeDelayEnd - TimeDelaySta
# print(TimeDelaySta )
# print(TimeDelayEnd )

conv_eeg_channel = 16
conv_audio_channel = 16
conv_time_size = 9
conv_output_channel = 10
fc_number = 30

conv_eeg_audio_number = 4
output_fc_number = conv_eeg_audio_number * conv_output_channel

# 模型选择
# True为CNN：D+S或CNN：FM+S模型，False为CNN：S模型
isDS = True
# isFM为0是男女信息，为1是方向信息
isFM = 0
# 频带数量
bands_number = 1
useFB = False

# 数据划分选择
# 测试集划分是否跨trail
isBeyoudTrail = False
# 是否使用100%的数据作为训练集，isBeyoudTrail=False、isALLTrain=True、need_pretrain = True、need_train = False说明跨被试
isALLTrain = False

# 预训练选择
# 只有train就是单独训练、只有pretrain是跨被试、两者都有是预训练
# 跨被试还需要上方的 isALLTrain 为 True
need_pretrain = False
need_train = True


# 整体模型


class CNN(nn.Module):
    def __init__(self, embed_dim=30,
                 num_heads=5,
                 layers=5,
                 attn_dropout=0.1,
                 relu_dropout=0.1,
                 res_dropout=0.1,
                 embed_dropout=0.25,
                 attn_mask=False):
        super(CNN, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.embed_dropout = embed_dropout
        self.attn_mask = attn_mask

        self.conv1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(conv_audio_channel, conv_output_channel, 8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        ) for i in range(conv_eeg_audio_number)])

        self.conv2 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(conv_audio_channel, conv_output_channel, 9),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        ) for i in range(conv_eeg_audio_number)])

        self.conv3 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(conv_audio_channel, conv_output_channel, 10),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        ) for i in range(conv_eeg_audio_number)])

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(fc_number, fc_number),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(fc_number, conv_output_channel)
        ) for i in range(conv_eeg_audio_number)])

        self.output_fc = nn.Sequential(
            nn.Linear(output_fc_number, output_fc_number), nn.Sigmoid(),
            nn.Linear(output_fc_number, 2), nn.Sigmoid()
        )

        self.proj_images = nn.Conv1d(eeg_channel, conv_eeg_channel, 1, padding=0, bias=False)
        self.proj_images2 = nn.Conv1d(eeg_channel, conv_eeg_channel, 1, padding=0, bias=False)
        self.proj_audio = nn.Conv1d(audio_channel, conv_audio_channel, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(audio_channel, conv_audio_channel, 1, bias=False)

        self.trans_a2e = self.get_network(self_type='a2e')
        self.trans_a2e2 = self.get_network(self_type='a2e')
        self.trans_e2a = self.get_network(self_type='e2a')
        self.trans_e2a2 = self.get_network(self_type='e2a')

    def get_network(self, self_type='a2e', layers=-1):
        embed_dim = 16
        num_heads = 1
        isAE = True

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  isAE=isAE)

    def cross_modal(self, x_audio, x_eeg, is_images=False, conv1D=False):
        '''
        audio and eeg should have dimension [batch_size, seq_len, n_features]
        if x_eeg is eegImages, the dimension should be [batch_size, seq_len, channel, height, width], default the images channel=1
        x_audio and x_eeg are both numpy array.
        '''

        x_audio = x_audio.transpose(1, 2)
        x_eeg = x_eeg.transpose(1, 2)

        #     proj_x_audio = self.proj_audio(x_audio)
        #     proj_x_eeg = self.proj_images(x_eeg)

        # proj_x_audio = proj_x_audio.permute(2, 0, 1)
        # proj_x_eeg = proj_x_eeg.permute(2, 0, 1)
        x_audio = x_audio.permute(2, 0, 1)
        x_eeg = x_eeg.permute(2, 0, 1)

        audio_trans_eeg, ae, ae_fc, ae_fc_last = self.trans_a2e(x_audio, x_eeg, x_eeg, True)
        audio_trans_eeg = audio_trans_eeg.permute(1, 0, 2)
        eeg_trans_audio, ea, ea_fc, ea_fc_last = self.trans_e2a(x_eeg, x_audio, x_audio, True)
        eeg_trans_audio = eeg_trans_audio.permute(1, 0, 2)

        return eeg_trans_audio, audio_trans_eeg, ae, ae_fc, ae_fc_last, ea, ea_fc, ea_fc_last

    def cross_modal2(self, x_audio, x_eeg, is_images=False, conv1D=False):
        '''
        audio and eeg should have dimension [batch_size, seq_len, n_features]
        if x_eeg is eegImages, the dimension should be [batch_size, seq_len, channel, height, width], default the images channel=1
        x_audio and x_eeg are both numpy array.
        '''

        x_audio = x_audio.transpose(1, 2)
        x_eeg = x_eeg.transpose(1, 2)

        #     proj_x_audio = self.proj_audio2(x_audio)
        #     proj_x_eeg = self.proj_images2(x_eeg)

        # proj_x_audio = proj_x_audio.permute(2, 0, 1)
        # proj_x_eeg = proj_x_eeg.permute(2, 0, 1)
        x_audio = x_audio.permute(2, 0, 1)
        x_eeg = x_eeg.permute(2, 0, 1)

        audio_trans_eeg, ae, ae_fc, ae_fc_last = self.trans_a2e2(x_audio, x_eeg, x_eeg, True)
        audio_trans_eeg = audio_trans_eeg.permute(1, 0, 2)
        eeg_trans_audio, ea, ea_fc, ea_fc_last = self.trans_e2a2(x_eeg, x_audio, x_audio, True)
        eeg_trans_audio = eeg_trans_audio.permute(1, 0, 2)

        return eeg_trans_audio, audio_trans_eeg, ae, ae_fc, ae_fc_last, ea, ea_fc, ea_fc_last

    def ChannelWav(self, wav, eeg):
        wav = wav.unsqueeze(0)

        Temp = torch.empty(((TimeDelayNum, window_length - TimeDelayEnd - abs(TimeDelaySta) - 1))).to(device)
        for k in range(TimeDelayNum):
            Temp[k, :] = wav[:, k + 1:-TimeDelayEnd - abs(TimeDelaySta) + k]

        wav = Temp.transpose(0, 1)
        weight = torch.matmul(eeg, wav).transpose(0, 1)
        weight = F.normalize(weight, p=2, dim=1)

        return wav, weight

    def WeightWav(self, wav, weight):
        weight = F.softmax(weight, dim=0)

        Temp = torch.ones(wav.shape[0], 1).to(device)
        Temp = torch.matmul(Temp, weight.transpose(0, 1))
        wav = wav.mul(Temp)
        # wav = wav.transpose(0,1).unsqueeze(0)
        wav = wav.unsqueeze(0)

        return wav

    def forward(self, x):

        wavA = x[0, 0, 0, :]
        wavB = x[0, 0, -1, :]
        eeg = x[0, 0, 1:-1, TimeDelayEnd:-abs(TimeDelaySta) - 1]

        # Time Delay 机制
        wavA, weightA = self.ChannelWav(wavA, eeg)
        wavB, weightB = self.ChannelWav(wavB, eeg)
        eeg = eeg.transpose(0, 1).unsqueeze(0)
        wavA = wavA.unsqueeze(0)
        wavB = wavB.unsqueeze(0)

        # print(x.shape)
        # print(wavA.shape)
        # print(eeg.shape)
        # print(wavB.shape)

        wavA_before = wavA.clone()

        eegA, wavA, aeA, ae_fcA, ae_fc_lastA, eaA, ea_fcA, ea_fc_lastA = self.cross_modal(wavA, eeg, is_images=False, conv1D=True)
        eegB, wavB, aeB, ae_fcB, ae_fc_lastB, eaB, ea_fcB, ea_fc_lastB = self.cross_modal2(wavB, eeg, is_images=False, conv1D=True)

        wavA_after = wavA.clone()

        # print("wavAB")
        # print(wavA.shape)
        # print(wavB.shape)

        wavA = wavA.transpose(1, 2)
        eegA = eegA.transpose(1, 2)
        eegB = eegB.transpose(1, 2)
        wavB = wavB.transpose(1, 2)

        # textCNN
        data = [wavA, eegA, eegB, wavB]
        for i in range(conv_eeg_audio_number):
            # print(wavA.shape)
            temp1 = self.conv1[i](data[i]).view(-1, conv_output_channel)
            # print(wavA1.shape)
            temp2 = self.conv2[i](data[i]).view(-1, conv_output_channel)
            # print(wavA2.shape)
            temp3 = self.conv3[i](data[i]).view(-1, conv_output_channel)
            # print(wavA3.shape)
            temp = torch.cat([temp1, temp2, temp3], dim=1)
            data[i] = self.fc[i](temp)

        x = torch.cat([data[i] for i in range(conv_eeg_audio_number)], dim=1)
        output = self.output_fc(x)
        return output, aeA, [wavA_before, wavA_after], ae_fcA, eaA, aeB, eaB, ae_fc_lastA


# 模型权重初始化
def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.fill_(0)


# 模型参数和初始化
myNet = CNN()
myNet.apply(weights_init_uniform)
clip = 0.8
optimzer = torch.optim.Adam(myNet.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimzer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
    min_lr=0, eps=0.001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max = 10, eta_min=0, last_epoch=-1)
loss_func = nn.CrossEntropyLoss()

# 启用gpu
device = torch.device('cuda:' + str(gpu_random))