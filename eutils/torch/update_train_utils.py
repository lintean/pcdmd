#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   update_train_utils.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/16 22:25   lintean      1.0         None
'''
import importlib

import numpy as np
import torch
from dotmap import DotMap
from torch.utils.data import DataLoader
import torch.nn.functional as func
from .container import DataLabelSet, EEGDataSet
from .util import load_model, save_model
from ..container import AADData, AADModel


def get_data_loader(data: AADData, args: DotMap, local: DotMap, **kwargs) -> tuple[AADData, DotMap, DotMap]:
    """
    获取数据集
    @param data:
    @param args:
    @param local:
    @return:
    """
    dev = data.aad_model.dev

    tng_set = EEGDataSet(eeg=data.eeg, audio=data.audio, window=data.tng_win, meta=data.meta, dev=dev)
    tes_set = EEGDataSet(eeg=data.eeg, audio=data.audio, window=data.tes_win, meta=data.meta, dev=dev)
    tng_loader = DataLoader(dataset=tng_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tes_loader = DataLoader(dataset=tes_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    data.tng_loader = tng_loader
    data.tes_loader = tes_loader

    return data, args, local


def run_func(aad_model: AADModel, dataloader: DataLoader, mode: str, local: DotMap, args: DotMap):
    with torch.set_grad_enabled(mode == "train"):
        losses, predict, targets = 0, [], []

        for (wav1, wav2, eeg, label, batch_idx) in dataloader:
            if len(label.shape) == 1:
                label = func.one_hot(label, num_classes=2).float()
            aad_model.optim.zero_grad()

            out, all_target, weights = aad_model.model(wav1, wav2, eeg, label)

            loss = aad_model.loss(out, all_target)
            losses = losses + loss.cpu().detach().numpy()

            # # # 是否需要可视化
            # if local.epoch in args.visualization_epoch and set(args.visualization_window_index) & set(window_index):
            #     for batch_index in range(len(window_index)):
            #         if (window_index[batch_index] in args.visualization_window_index):
            #             for index in range(len(weights)):
            #                 temp = DotMap(weights[index])
            #                 temp.data = weights[index].data[batch_index]
            #                 util.heatmap(temp, local.log_path, window_index[batch_index], local.epoch, local.name)

            predict.append(out)
            targets.append(all_target)
            if mode == "train":
                loss.backward()
                aad_model.optim.step()
                if aad_model.warmup is not None:
                    with aad_model.warmup.dampening():
                        pass

        average_loss = losses / len(dataloader)
        output = (predict, targets, average_loss) if mode == "test" else average_loss
        return output


def trainer(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    torch.set_num_threads(1)
    tng_loader, tes_loader, aad_model = data.tng_loader, data.tes_loader, data.aad_model

    for epoch in range(local.max_epoch):
        local.epoch = epoch

        # 训练和验证
        aad_model.model.train()
        tng_loader.dataset.call_epoch()
        loss_train = run_func(aad_model, tng_loader, "train", local, args)
        aad_model.model.eval()
        # loss = modeling(data, vali_window, "vali", window_metadata, False, local, args)
        loss = 0
        loss2 = run_func(aad_model, tes_loader, "vali", local, args)

        # 学习率衰减
        if aad_model.sched is not None:
            if aad_model.warmup is not None:
                with aad_model.warmup.dampening():
                    aad_model.sched.step()
            else:
                aad_model.sched.step()

        current_lr = aad_model.optim.state_dict()['param_groups'][0]['lr']
        local.logger.info(f"{epoch} "
                          f"train:{round(loss_train, 6)} "
                          f"valid:{round(loss, 6)} "
                          f"test:{round(loss2, 6)} "
                          f"lr:{round(current_lr, 6)}")
    return data, args, local


def tester(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    tng_loader, tes_loader, aad_model = data.tng_loader, data.tes_loader, data.aad_model
    total_t_num = 0
    total_f_num = 0
    aad_model.model.eval()
    for num in range(1):
        t_num, f_num = 0, 0
        predict, targets, ave_loss = run_func(aad_model, tes_loader, "test", local, args)

        p = torch.concat(predict, dim=0)
        t = torch.concat(targets, dim=0)
        p = func.softmax(p, dim=1)
        p, t = p.cpu().detach().numpy(), t.cpu().detach().numpy()
        judge = []
        for i in range(p.shape[0]):
            if (p[i, 0] > 0.5 and t[i, 0] == 1) or (p[i, 1] > 0.5 and t[i, 1] == 1):
                t_num += 1
                judge.append(1)
            else:
                f_num += 1
                judge.append(0)
        # real = np.array([(t[i, 0] == 1) for i in range(t.shape[0])])[:, None]
        # pred = np.array([(p[i, 0] > 0.5) for i in range(p.shape[0])])[:, None]
        # log = tes_loader.dataset.window.cpu().detach().numpy()
        # log = np.concatenate([log, real, pred, np.array(judge)[:, None]], axis=1)
        # result_logging(local.log_path, local.name, log)
        local.logger.info(f'first detect true:{t_num} false:{f_num}')

        total_t_num = total_t_num + t_num
        total_f_num = total_f_num + f_num
    local.logger.info("\n" + str(total_t_num / (total_t_num + total_f_num)))
    return data, args, local


def save(data: AADData, args: DotMap, local: DotMap) -> tuple[AADData, DotMap, DotMap]:
    save_model(aad_model=data.aad_model, local=local, seed=args.random_seed)
    return data, args, local


def get_model(data: AADData, args: DotMap, local: DotMap)\
        -> tuple[AADData, DotMap, DotMap]:

    module = importlib.import_module(f"{local.model_path}")
    aad_model = module.get_model(local)
    local.dev = aad_model.dev
    local.logger.info(aad_model.dev)
    data.aad_model = aad_model

    local.logger.info(id(aad_model.model))
    return data, args, local


