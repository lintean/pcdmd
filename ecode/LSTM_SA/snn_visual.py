#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   snn_power.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/23 19:53   lintean      1.0         None
'''
import math
import importlib
import time

import torch

from eutils import util
from eutils.util import makePath
from dotmap import DotMap
import logging
from eutils.torch.util import load_model
from eutils.torch.train_utils import get_data_loader
import torch.nn.functional as func
from multiprocessing import Process


def get_logger(log_path):
    # 第一步，创建一个logger
    logger = logging.getLogger(f"snn_power_logger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 第二步，创建一个handler，用于写入日志文件
    logfile = f"{makePath(log_path)}/snn_power_logger.log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)


    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def modeling(dataloader, mode, local, args):
    local.aad_model.model.eval()
    losses = 0
    with torch.no_grad():
        for (wav1, wav2, eeg, label, batch_idx) in dataloader:
            if batch_idx * 2 > len(dataloader): break
            if len(label.shape) == 1:
                label = func.one_hot(label, num_classes=2).float()
            local.aad_model.optim.zero_grad()

            out, all_target, weights = local.aad_model.model(wav1, wav2, eeg, label)

            loss = local.aad_model.loss(out, all_target)
            losses = losses + loss.cpu().detach().numpy()

            label = None
            t, p = all_target, out
            p = func.softmax(p, dim=1)
            if p[0, 0] > 0.5 and t[0, 0] == 1:
                label = 1
            elif p[0, 1] > 0.5 and t[0, 0] == 1:
                label = 2
            elif p[0, 0] > 0.5 and t[0, 1] == 1:
                label = 3
            elif p[0, 1] > 0.5 and t[0, 1] == 1:
                label = 4

            # 是否需要可视化
            for index in range(len(weights)):
                util.heatmap(weights[index], local.log_path, int(batch_idx), name=local.name, label=label)

    local.logger.info(losses / len(dataloader))



def main(local: DotMap = DotMap(), args: DotMap = DotMap()):
    torch.set_num_threads(1)
    if not args:
        module = importlib.import_module(f"parameters")
        args = module.args

    log_path = f"{device_to_use.project_root_path}/result/BSAnet"
    logger = get_logger(log_path)
    logger.setLevel(logging.WARNING)

    model_type = "RSNN"
    model = f"models.LSTM_SA.lstm_v3_visual"
    database = "KUL" if "KUL" in args.data_document_path else "DTU"
    win_lens = {"1": 1}

    if model_type == "RSNN" or model_type == "SSA":
        flod_num = 5
        names = args.names
    else:
        flod_num = 1
        names = ["S1"]
    flods = [f"flod_{i}" for i in range(flod_num)]

    for len_str, win_len in win_lens.items():
        for flod in flods:
            process = []
            for name in names:

                args.window_length = math.ceil(args.data_meta.fs * win_len)
                args.batch_size = 1
                args.current_flod = int(flod[-1])
                args.snn_process = model_type == "RSNN" or model_type == "SSA"
                args.need_sa = "SA" in model_type

                local.name = name
                local.name_index = int(local.name[1:]) - 1
                local.log_path = f"{log_path}/{model_type}/{database}_{len_str}/{flod}/pic"
                local.logger = logger

                module = importlib.import_module(model)
                aad_model = module.get_model(args)

                if model_type == "RSNN" or model_type == "SSA":
                    file_name = f"{log_path}/{model_type}/{database}_{len_str}/{flod}/model-{name}.pth.tar"
                    aad_model = load_model(file_name=file_name, aad_model=aad_model)

                local.aad_model = aad_model

                (tng_loader, tes_loader), args, local = get_data_loader(local, args)
                # modeling(tes_loader, "test", local, args)

                p = Process(target=modeling, args=(tes_loader, "test", local, args))  # 必须加,号
                p.start()
                local.logger.info(f"开始{name}线程")
                time.sleep(20)
                process.append(p)

            for p in process:
                p.join()



if __name__ == "__main__":
    main()
