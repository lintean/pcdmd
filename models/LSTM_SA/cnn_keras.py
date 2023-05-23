from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, losses
import eutils.split_utils as sutil
import math
import eutils.util as util
from dotmap import DotMap
import time
import tensorflow as tf

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_document_path = device_to_use.origin_data_document + "/KUL_single_single_snn_1to32_mean"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "cnn_keras"
args.ConType = ["No"]

# 加载数据集元数据
data_meta = util.read_json(args.data_document_path + "/metadata.json")
args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
# args.names = ["S10"]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
args.window_length = math.ceil(data_meta.fs * 1)
# args.window_lap = math.ceil(data_meta.fs * 0.5)
args.window_lap = None
args.overlap = 1 - args.window_lap / args.window_length if args.window_lap is not None else 0.5
args.delay = 0
args.batch_size = 32
args.max_epoch = 100
args.min_epoch = 0
args.lr = 1e-3
args.early_patience = 0
args.random_seed = time.time()
args.cross_validation_fold = 5
args.current_flod = 0
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
args.process_steps = [sutil.read_prepared_data, sutil.subject_split]

model = Sequential()
model.add(keras.layers.Permute((1, 2), input_shape=(args.window_length, args.eeg_channel)))
model.add(keras.layers.Conv1D(filters=10, kernel_size=9, activation='relu'))
model.add(keras.layers.GlobalAvgPool1D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(2, activation='sigmoid'))

model.summary()

# 模型参数和初始化
clip = 0.8
optimzer = optimizers.Adam(args.lr)
scheduler = None
loss_func = losses.BinaryCrossentropy()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                                                       patience=10, min_delta=1e-3)
