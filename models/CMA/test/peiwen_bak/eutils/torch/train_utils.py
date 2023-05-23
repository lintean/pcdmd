#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_utils.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/17 10:19   lintean      1.0         None
'''
import numpy as np
import torch
from dotmap import DotMap
from torch.utils.data import DataLoader
import torch.nn.functional as func

from split import split_data
from . import util
from .container import CurrSet, DataLabelSet
from .util import load_model
from eutils.util import result_logging


def to_data_label(data, window, wm, args: DotMap, local: DotMap):
    wav1 = np.empty((window.shape[0], args.audio_channel, args.window_length), dtype=np.float32)
    wav2 = np.empty((window.shape[0], args.audio_channel, args.window_length), dtype=np.float32)
    eeg = np.empty((window.shape[0], args.eeg_channel, args.window_length), dtype=np.float32)
    label = np.empty(window.shape[0], dtype=np.int64)
    for i in range(window.shape[0]):
        wav1[i, ...] = data[0][window[i, wm.start]: window[i, wm.end]].transpose(-1, -2)
        wav2[i, ...] = data[1][window[i, wm.start]: window[i, wm.end]].transpose(-1, -2)
        eeg[i, ...] = data[2][window[i, wm.start]: window[i, wm.end]].transpose(-1, -2)
        label[i] = window[i, wm.target] - 1

    dev = local.aad_model.dev
    wav1 = torch.from_numpy(wav1).to(dev)
    wav2 = torch.from_numpy(wav2).to(dev)
    eeg = torch.from_numpy(eeg).to(dev)
    label = torch.from_numpy(label).to(dev)
    label = func.one_hot(label, num_classes=2).float()
    return wav1, wav2, eeg, label


def get_data_label_loader(local: DotMap = DotMap(), args: DotMap = DotMap(), **kwargs):
    """
    获取数据集
    @param args:
    @param local:
    @return:
    """
    data, train_window, test_window, window_metadata = split_data(local.name, log_path=local.log_path, args=args, local=local)
    window_metadata = DotMap(window_metadata)

    wav1, wav2, eeg, label = to_data_label(data, train_window, window_metadata, args, local)
    tng_set = DataLabelSet(eeg=eeg, wav1=wav1, wav2=wav2, label=label)
    wav1, wav2, eeg, label = to_data_label(data, test_window, window_metadata, args, local)
    tes_set = DataLabelSet(eeg=eeg, wav1=wav1, wav2=wav2, label=label)
    tng_loader = DataLoader(dataset=tng_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    tes_loader = DataLoader(dataset=tes_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    return (tng_loader, tes_loader), args, local


def get_data_loader(local: DotMap = DotMap(), args: DotMap = DotMap(), **kwargs):
    """
    获取数据集
    @param args:
    @param local:
    @return:
    """
    data, train_window, test_window, window_metadata = split_data(local.name, log_path=local.log_path, args=args, local=local)
    window_metadata = DotMap(window_metadata)

    dev = local.aad_model.dev
    if data[0].dtype == np.float32:
        wav1 = torch.from_numpy(data[0].transpose(-1, -2)).to(dev)
        wav2 = torch.from_numpy(data[1].transpose(-1, -2)).to(dev)
        eeg = torch.from_numpy(data[2].transpose(-1, -2)).to(dev)
    else:
        wav1 = torch.tensor(data[0].transpose(-1, -2), dtype=torch.float32, device=dev)
        wav2 = torch.tensor(data[1].transpose(-1, -2), dtype=torch.float32, device=dev)
        eeg = torch.tensor(data[2].transpose(-1, -2), dtype=torch.float32, device=dev)

    if train_window.dtype == np.int64:
        train_window = torch.from_numpy(train_window).to(dev)
        test_window = torch.from_numpy(test_window).to(dev)
    else:
        train_window = torch.tensor(train_window, dtype=torch.int64, device=dev)
        test_window = torch.tensor(test_window, dtype=torch.int64, device=dev)

    tng_set = CurrSet(eeg=eeg, wav1=wav1, wav2=wav2, window=train_window, window_meta=window_metadata, args=args)
    tes_set = CurrSet(eeg=eeg, wav1=wav1, wav2=wav2, window=test_window, window_meta=window_metadata, args=args)
    tng_loader = DataLoader(dataset=tng_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    tes_loader = DataLoader(dataset=tes_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    return (tng_loader, tes_loader), args, local


def modeling(dataloader, mode, local, args):
    with torch.set_grad_enabled(mode == "train"):
        losses, predict, targets = 0, [], []

        for (wav1, wav2, eeg, label, batch_idx) in dataloader:
            if len(label.shape) == 1:
                label = func.one_hot(label, num_classes=2).float()
            local.aad_model.optim.zero_grad()

            out, all_target, weights = local.aad_model.model(wav1, wav2, eeg, label)

            loss = local.aad_model.loss(out, all_target)
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
                local.aad_model.optim.step()

        average_loss = losses / len(dataloader)
        output = (predict, targets, average_loss) if mode == "test" else average_loss
        return output


def trainEpoch(data_loader, local, args):
    tng_loader, tes_loader = data_loader

    for epoch in range(args.max_epoch):
        local.epoch = epoch

        # 训练和验证
        local.aad_model.model.train()
        tng_loader.dataset.call_epoch()
        loss_train = modeling(tng_loader, "train", local, args)
        local.aad_model.model.eval()
        # loss = modeling(data, vali_window, "vali", window_metadata, False, local, args)
        loss = 0
        loss2 = modeling(tes_loader, "vali", local, args)

        # 学习率衰减
        if local.aad_model.sched:
            local.aad_model.sched.step()
        # local.scheduler.step(loss)
        # parameters.scheduler.step(0.1)

        local.logger.info(f"{epoch} {loss_train} {loss} {loss2}")


def testEpoch(data_loader: tuple, local: DotMap = DotMap(), args: DotMap = DotMap()):
    tng_loader, tes_loader = data_loader
    total_t_num = 0
    total_f_num = 0
    local.aad_model.model.eval()
    for num in range(1):
        t_num, f_num = 0, 0
        predict, targets, ave_loss = modeling(tes_loader, "test", local, args)

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
        real = np.array([(t[i, 0] == 1) for i in range(t.shape[0])])[:, None]
        pred = np.array([(p[i, 0] > 0.5) for i in range(p.shape[0])])[:, None]
        log = tes_loader.dataset.window.cpu().detach().numpy()
        log = np.concatenate([log, real, pred, np.array(judge)[:, None]], axis=1)
        result_logging(local.log_path, local.name, log)
        local.logger.info(f'first detect true:{t_num} false:{f_num}')

        total_t_num = total_t_num + t_num
        total_f_num = total_f_num + f_num
    local.logger.info("\n" + str(total_t_num / (total_t_num + total_f_num)))
    return None, args, local


def single_model(data_loader: tuple, local: DotMap = DotMap(), args: DotMap = DotMap()):
    torch.set_num_threads(1)
    trainEpoch(data_loader, local, args)
    util.save_model(aad_model=local.aad_model, local=local, seed=args.random_seed)
    # 测试最后一个模型
    local.logger.info("test:")
    testEpoch(data_loader, local, args)

    return None, args, local


def load(data_loader: tuple, local: DotMap = DotMap(), args: DotMap = DotMap()):
    model_file = f"{args.load_model_path}/flod_{args.current_flod}/model-{local.name}.pth.tar"
    local.aad_model, random_seed = load_model(file_name=model_file, aad_model=local.aad_model, map_location=local.aad_model.dev)

    return data_loader, args, local