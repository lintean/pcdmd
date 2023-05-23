import eutils.util as util
import eutils.split_utils as sutil
from dotmap import DotMap
import sys
import numpy as np
import shutil


def split_data(name="S1", log_path="./result/test", args=None, local=None):
    # 设定流程：
    # 1.读取数据
    # 2.划分时间窗口并生成训练集和测试集
    process_steps = [sutil.read_prepared_data, sutil.subject_split]
    if "process_steps" in args:
        process_steps = args.process_steps
    # process_steps = [sutil.get_data_from_preprocess, sutil.window_split]
    # process_steps = [util.window_split_new]

    # 执行流程
    # 构造局部命名空间
    _local = DotMap() if local is None else local
    _local.name = name
    _local.subject_number = int(_local.name[1:])
    _local.logger.info("overlap: " + str(args.overlap))
    data_temp = None

    for j in range(len(process_steps)):
        _local.logger.info("working process: " + process_steps[j].__name__)
        data_temp, args, _local = process_steps[j](data=data_temp, args=args, local=_local)


    # 保存数据
    data = data_temp[0]
    train = data_temp[1]
    test = data_temp[2]
    window_metadata = dict(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5, reversed=6)
    if "need_save" in _local and _local.need_save:
        np.savez(util.makePath(args.data_document) + "/" + name, data=data, train_window=train, test_window=test,
             window_metadata=window_metadata)
        shutil.copy(args.data_document_path + "/metadata.json", args.data_document)
        _local.logger.info("working process: saving date in " + args.data_document + "/" + name + ".npz")

    return data, train, test, window_metadata


if __name__ == "__main__":
    if (len(sys.argv) >= 3):
        split_data(sys.argv[1], sys.argv[2])
    else:
        split_data()
