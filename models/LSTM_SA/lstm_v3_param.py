import math
import eutils.util as util
import eutils.split_utils as sutil
from dotmap import DotMap
import time
from eutils.torch.train_utils import get_data_loader, single_model

# metadata字典
args = DotMap()

# 所用的数据目录路径
# args.data_document_path = device_to_use.origin_data_document + "/DTU_single_single_snn_1to32"
args.data_document_path = device_to_use.origin_data_document + "/KUL_single_single_snn_1to32_mean"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "LSTM_final"
args.ConType = ["No"]
data_meta = util.read_json(args.data_document_path + "/metadata.json")
args.data_meta = data_meta
args.names = [f"S{i + 1}" for i in range(data_meta.people_number)]
# args.names = ["S14"]

# 模型相关参数
args.model_path = f"models.LSTM_SA.lstm_v3"
args.need_sa = True
args.snn_process = False
args.vth = 0.5
# args.tau_mem = math.exp(-1 / 128 / 0.005)
# args.tau_syn = math.exp(-1 / 128 / 0.005)
args.tau_mem = 0.25
args.tau_syn = 0.25
print(args.tau_mem, " ", args.tau_syn)

# 处理步骤
args.process_steps = [sutil.read_prepared_data, sutil.subject_split] if "KUL" in args.data_document_path \
    else [sutil.get_data_from_preprocess, sutil.subject_split]
args.process_steps += [sutil.remove_repeated]
args.train_steps = [get_data_loader, single_model]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
args.window_length = math.ceil(data_meta.fs * 2)
args.window_lap = math.ceil(data_meta.fs * 0.2) if "DTU" in args.data_document_path else math.ceil(data_meta.fs * 0.5)
# args.window_lap = None
args.overlap = 1 - args.window_lap / args.window_length if args.window_lap is not None else 0
args.delay = 0
args.batch_size = 32
args.max_epoch = 50
args.lr = 1e-3
args.early_patience = 0
args.random_seed = time.time()
args.cross_validation_fold = 5
# args.current_flod = 0

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

# DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
args.isFM = 0 if "KUL" in args.data_document_path else 1