#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/17 15:04   lintean      1.0         None
'''

import sys
import importlib
import ecfg
from eutils.util import makePath
from dotmap import DotMap
import logging


def get_logger(name, log_path):
    # 第一步，创建一个logger
    logger = logging.getLogger(f"train_{name}_logger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 第二步，创建一个handler，用于写入日志文件
    logfile = f"{makePath(log_path)}/Train_{name}.log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)

    if log_path == "./result/test":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def init(name, log_path, local, args):
    logger = get_logger(name, log_path)

    if not args:
        module = importlib.import_module(f"parameters")
        args = module.args
        logger.info(f"read parameters from ./parameters.py")

    local.name = name
    local.log_path = log_path
    local.logger = logger
    local.lr = args.lr
    local.batch_size = args.batch_size
    local.max_epoch = args.max_epoch
    local.split_meta = args.split_meta
    local.model_meta = args.model_meta
    local.model_path = args.model_path
    local.preproc_meta = args.preproc_meta

    local.logger.info(f'data path: {args.data_document_path}')
    args.gpu_list = ecfg.gpu_list
    return local, args


def main(name: str = "S1", log_path: str = "./result/test", local: DotMap = DotMap(), args: DotMap = None):
    local, args = init(name, log_path, local, args)

    # 设定流程：
    # 1.读取数据生成dataloader
    # 2.训练并测试模型
    assert "proc_steps" in args, f"cannot find proc_steps in args"

    data = None
    for proc in args.proc_steps:
        local.logger.info(f"working process: {proc.__name__}")
        data, args, local = proc(data=data, args=args, local=local)


if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1])
    else:
        main()
