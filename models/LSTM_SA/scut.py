import math
import ecfg as cfg
import eutils.util as util
from eutils.update_split_utils import *
from dotmap import DotMap
import time
from eutils.torch.update_train_utils import *
from eutils.container import PreprocMeta
import db

# metadata字典
args = DotMap()

# 所用的数据目录路径
args.data_name = "/SCUT_test"
args.data_document_path = cfg.origin_data_document + args.data_name
args.database = db.get_db_from_name(args.data_name)

# 输入数据选择
# label 为该次训练的标识
# ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
# names 为这次训练用到的被试数据
args.label = "LSTM_final"
args.ConType = ["No"]
args.names = [f"S{i + 1}" for i in range(args.database.subj_number)]
args.random_seed = time.time()

# 模型相关参数
args.model_path = f"models.LSTM_SA.lstm_v3"
args.model_meta = DotMap(
    need_sa=False,
    snn_process=True,
    vth=0.5,
    tau_mem=0.25,
    tau_syn=0.25
)

# 处理步骤
args.proc_steps = [
    preproc, trails_split, hold_on_divide,
    get_model, get_data_loader, trainer, save, tester
]

# 常用模型参数，分别是 重复率、窗长、时延、最大迭代次数、分批训练参数、是否early stop
args.delay = 0
args.batch_size = 32
args.max_epoch = 50
args.lr = 1e-3
args.early_patience = 0

# preproc meta
args.preproc_meta = PreprocMeta(
    eeg_lf=1,
    eeg_hf=32,
    wav_lf=1,
    wav_hf=32,
    label_type="direction",
    need_voice=False,
    ica=True
)
# DTU:0是男女信息，1是方向信息; KUL:0是方向信息，1是人物信息
# args.isFM = 0 if "KUL" in args.data_document_path else 1

# split meta
tl = 2
args.split_meta = SplitMeta(
    time_len=tl,
    time_lap=0.2,
    # overlap=0 if tl < 0.5 else None,
    tes_pct=0.2,
    valid_pct=0
)

# 可视化选项 列表为空表示不希望可视化
args.visualization_epoch = []
args.visualization_window_index = []

