import math
import ecfg as cfg
import eutils.util as util
from eutils.split import *
from dotmap import DotMap
import time
from eutils.torch.train import *
from eutils.container import PreprocMeta
import db

"""
全局参数容器
"""
args = DotMap()

"""
输入数据参数

args.data_name为读取的数据目录名
args.data_document_path不需要设置，为读取的数据目录路径
args.database不需要设置，会自动根据数据目录名判断是哪个数据库，因此数据目录名必须带有数据库名称

args.label 为该次训练的标识
args.ConType 为选用数据的声学环境，如果ConType = ["No", "Low", "High"]，则将三种声学数据混合在一起后进行训练
args.names 一般需要设置，是一个数组，包含 multiple_train 一次训练需要跑的被试。如果args.names=['S1']则multiple_train仅会跑第一个被试
args.random_seed 是该次训练所使用的随机种子
"""
args.data_name = "/SCUT_test"
args.data_document_path = cfg.origin_data_document + args.data_name
args.database = db.get_db_from_name(args.data_name)

args.label = "multiband"
args.ConType = ["No"]
args.names = [f"S{i + 1}" for i in range(args.database.subj_number)]
args.random_seed = time.time()

"""
模型相关参数

args.model_path 为该次训练所使用模型。args.model_path = "models.CNN.CNN"表示使用项目目录下models/CNN/CNN.py的模型进行训练
args.model_meta 为需要传递给模型初始化的参数。默认为空
"""
args.model_path = f"models.multi-band.multiband_update"

"""
定义训练流程

args.proc_steps 为该次训练（包含测试）的流程。是一个数组，包含一系列函数名。训练会顺序执行里面的函数，以完成训练过程
更改args.proc_steps可以改变训练（包含测试）的流程
"""
args.proc_steps = [
    preproc, trails_split, hold_on_divide,
    get_model, get_data_loader, trainer, save, tester
]

"""
训练相关参数

args.batch_size 为批大小
args.max_epoch 为最大迭代次数。args.max_epoch = 100代表训练会在达到100次epoch后停止
args.lr 为学习率
args.early_patience 为early stop参数。注：因版本迭代，early stop代码已丢失，需手动实现。
"""
args.batch_size = 16
args.max_epoch = 100
args.lr = 2e-4
args.early_patience = 0

"""
预处理/读取数据的参数

args.preproc_meta 为PreprocMeta结构，里面的参数不需要全部给出。
如果是读取已预处理的数据，只需要给出need_voice（是否需要语音）、label_type（数据label是方位还是语音），如：
    args.preproc_meta = PreprocMeta(
        need_voice=False,
        label_type="direction"
    )
如果是预处理数据，需要给出预处理相关参数。如：
    args.preproc_meta = PreprocMeta(
        eeg_lf=1,
        eeg_hf=32,
        wav_lf=1,
        wav_hf=32,
        label_type="direction",
        need_voice=False,
        ica=True
    )
"""
args.preproc_meta = PreprocMeta(
    eeg_lf=[1],
    eeg_hf=[50],
    wav_lf=1,
    wav_hf=50,
    label_type="speaker",
    need_voice=True,
    ica=True,
    internal_fs=8000,
    gl=100,
    gh=8000,
    space=3,
    is_combine=False
)

"""
划分窗口、划分数据集的参数

args.split_meta 为SplitMeta结构，里面的参数不需要全部给出。
    args.split_meta = SplitMeta(
        time_len=1,
        overlap=0,
        cv_flod=5,
        curr_flod=0,
        tes_pct=0.2,
        valid_pct=0
    )
"""
args.split_meta = SplitMeta(
    time_len=0.25,
    time_lap=0.25,
    # overlap=0 if tl < 0.5 else None,
    tes_pct=0.2,
    valid_pct=0
)

"""
用于可视化的参数

这部分代码可能需要手动修改
"""
args.visualization_epoch = []
args.visualization_window_index = []

