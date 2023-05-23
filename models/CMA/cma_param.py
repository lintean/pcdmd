import eutils.util as util
import time
from eutils.split_utils import *
from eutils.torch.train_utils import get_data_loader, single_model

# metadata字典
args = DotMap()

# 所用的数据目录路径
# args.data_document_path = device_to_use.origin_data_document + "/KUL_single_single_clean_1to32_new"
# args.data_document_path = device_to_use.origin_data_document + "/KUL_single_single_separate_1to32_new"
args.data_document_path = device_to_use.origin_data_document + "/DTU_single_single_snn_1to32"
# args.data_document_path = device_to_use.origin_data_document + "/KUL_single_single_separate_1to32_new"
args.load_model_path = f"{device_to_use.project_root_path}/result/cma_Encoder_clean_speech_2023-02-08_20_42_53"

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "cma_Encoder_clean_speech"
args.ConType = ["No", "Low", "High"]

# 加载数据集元数据
data_meta = util.read_json(args.data_document_path + "/metadata.json")
args.data_meta = data_meta
args.names = ["S" + str(i + 1) for i in range(data_meta.people_number)]
# args.names = ["S10"]

# 模型相关参数
fs = 8000
args.data_meta.audio_fs, data_meta.audio_fs, args.audio_fs = fs, fs, fs
args.model_path = f"models.CMA.cma_plus"
args.extra_audio = "separate" if "separate" in args.data_document_path else "origin"
args.encoder = False
args.cma_layer = 1
args.random_reverse = False
args.need_hrtf = True
args.hrtf_transform = True
args.only_1and2 = False
args.transformer = False
args.l2 = 0
args.ica = True

args.q_act, args.k_act, args.v_act = True, True, True
args.cmaout, args.cmaout_act = False, False

args.h = 16
args.input_size = 64
args.cls_channel = 32
args.audio_conv_size = 17
args.eeg_conv_size = 17

def cont_judge(args, left_file, right_file) -> bool:
    if ("need_hrtf" in args and not args.need_hrtf) and ("hrtf" in left_file or "hrtf" in right_file):
        return True
    if ("only_1and2" in args and args.only_1and2) and ("3" in left_file or "4" in right_file):
        return True
    return False

args.cont_judge = cont_judge

def output(data_loader: tuple, local: DotMap = DotMap(), args: DotMap = DotMap()):
    arr = local.aad_model.model.conv3.weight.cpu().detach().numpy()
    np.save(file=f"{args.load_model_path}/{local.name}_encoder", arr=arr)
    return data_loader, local, args

# 处理步骤
if args.encoder:
    args.process_steps = [read_extra_audio]
else:
    # args.process_steps = [read_eeg_wav] if "separate" in args.data_document_path else [read_cma_data]
    if args.ica:
        args.process_steps = [preprocess_eeg_audio]
    else:
        args.process_steps = [read_eeg_wav]
if "DTU" in args.data_document_path:
    args.process_steps = [read_prepared_data]
args.process_steps += [subject_split, remove_repeated, add_negative_samples]
args.train_steps = [get_data_loader, single_model]
# args.train_steps = [load, get_data_loader, testEpoch]
# args.train_steps = [load, get_data_loader, output]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
# 其中窗长和时延，因为采样率为70hz，所以70为1秒
args.time_length = 1
args.window_length = math.ceil(data_meta.fs * args.time_length)
args.window_lap = math.ceil(data_meta.fs * 0.2)
# args.window_lap = None
args.overlap = 1 - args.window_lap / args.window_length if args.window_lap is not None else 0.5
args.delay = 0
args.batch_size = 8
args.max_epoch = 100
args.lr = 1e-3
args.early_patience = 0
args.random_seed = int(time.time())
args.cross_validation_fold = 5
args.current_flod = 0

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
args.trail_number = 24
args.cell_number = data_meta.cell_number
args.bands_number = data_meta.bands_number
args.fs = data_meta.fs
args.test_percent = 0.2
args.vali_percent = 0

# DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
args.isFM = 1
# 先验 "default" “none” "speaker" "direction"
args.prior = "default"