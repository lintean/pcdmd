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

import numpy as np
import torch

from eutils.util import makePath
from dotmap import DotMap
import logging
from ecfg import origin_data_document
from eutils.torch.util import load_model
from eutils.torch.train_utils import get_data_loader
import torch.nn.functional as func
import pandas as pd

from eutils.snn.container import SPower


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
    energys = SPower()
    with torch.no_grad():
        for (wav1, wav2, eeg, label, batch_idx) in dataloader:
            if len(label.shape) == 1:
                label = func.one_hot(label, num_classes=2).float()
            local.aad_model.optim.zero_grad()

            out, all_target, weights = local.aad_model.model(wav1, wav2, eeg, label)

            loss = local.aad_model.loss(out, all_target)
            losses = losses + loss.cpu().detach().numpy()

            energys += weights

            # # # 是否需要可视化
            # if local.epoch in args.visualization_epoch and set(args.visualization_window_index) & set(window_index):
            #     for batch_index in range(len(window_index)):
            #         if (window_index[batch_index] in args.visualization_window_index):
            #             for index in range(len(weights)):
            #                 temp = DotMap(weights[index])
            #                 temp.data = weights[index].data[batch_index]
            #                 util.heatmap(temp, local.log_path, window_index[batch_index], local.epoch, local.name)

    return losses / len(dataloader), energys * (1.0 / len(dataloader))



def main(local: DotMap = DotMap(), args: DotMap = DotMap()):
    torch.set_num_threads(1)
    if not args:
        module = importlib.import_module(f"parameters")
        args = module.args

    log_path = f"{device_to_use.project_root_path}/result/BSAnet"
    logger = get_logger(log_path)
    logger.setLevel(logging.WARNING)

    model_type = "SSA"
    model = f"models.LSTM_SA.lstm_v3_power"
    database = "KUL"
    win_lens = {"025": 0.25}

    if model_type == "RSNN" or model_type == "SSA":
        flod_num = 5
        names = args.names
    else:
        flod_num = 1
        names = ["S1"]
    flods = [f"flod_{i}" for i in range(flod_num)]

    e, mac, spikes, firing_rate, param = [], [], [], [], []
    sep = np.stack([np.zeros(shape=args.people_number), np.array(args.names)], axis=0)
    sep = pd.DataFrame(sep)

    for len_str, win_len in win_lens.items():
        result = []
        for flod in flods:
            e_subject = []
            for name in names:

                args.window_length = math.ceil(args.data_meta.fs * win_len)
                if database == "KUL":
                    args.data_document_path = f"{origin_data_document}/KUL_single_single_snn_1to32_mean"
                    args.overlap = 0.5
                else:
                    args.data_document_path = f"{origin_data_document}/DTU_single_single_snn_1to32"
                    args.overlap = 0.2

                args.batch_size = 1
                args.current_flod = int(flod[-1])
                args.snn_process = model_type == "RSNN" or model_type == "SSA"
                args.need_sa = "SA" in model_type

                local.name = name
                local.name_index = int(local.name[1:]) - 1
                local.log_path = log_path
                local.logger = logger

                module = importlib.import_module(model)
                aad_model = module.get_model(args)

                if model_type == "RSNN" or model_type == "SSA":
                    file_name = f"{log_path}/{model_type}/{database}_{len_str}/{flod}/model-{name}.pth.tar"
                    aad_model = load_model(file_name=file_name, aad_model=aad_model)
                    if isinstance(aad_model, tuple):
                        aad_model, args.random_seed = aad_model

                local.aad_model = aad_model

                (tng_loader, tes_loader), args, local = get_data_loader(local, args)
                # testEpoch(tes_loader, local, args)
                loss, energy = modeling(tes_loader, "test", local, args)

                local.logger.warning(loss)
                e_subject.append(energy.to_numpy())

            result.append(e_subject)

        result = np.array(result)
        e.append(pd.DataFrame(result[..., 0]))
        e.append(sep)
        mac.append(pd.DataFrame(result[..., 1]))
        mac.append(sep)
        spikes.append(pd.DataFrame(result[..., 2]))
        spikes.append(sep)
        firing_rate.append(pd.DataFrame(result[..., 2] / result[..., 3]))
        firing_rate.append(sep)


    e = pd.concat(e, ignore_index=True).T
    mac = pd.concat(mac, ignore_index=True).T
    spikes = pd.concat(spikes, ignore_index=True).T
    firing_rate = pd.concat(firing_rate, ignore_index=True).T

    save_path = makePath(f"{log_path}/{model_type}")
    e.to_csv(f"{save_path}/{database}_energy.csv", index=False, header=False)
    mac.to_csv(f"{save_path}/{database}_mac.csv", index=False, header=False)
    spikes.to_csv(f"{save_path}/{database}_spikes.csv", index=False, header=False)
    firing_rate.to_csv(f"{save_path}/{database}_firing_rate.csv", index=False, header=False)


if __name__ == "__main__":
    main()
