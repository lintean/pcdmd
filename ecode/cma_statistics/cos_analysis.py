#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cos_analysis.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/26 20:46   lintean      1.0         None
'''
import numpy as np
import torch
import torch.nn.functional as func
from train import init
from eutils.split_utils import *
from eutils.torch.train_utils import get_data_loader, single_model, load, testEpoch
from matplotlib import pyplot as plt


def get_cos(dataloader, local, args, mode="test"):
    cos = []
    with torch.set_grad_enabled(mode == "train"):
        losses, predict, targets = 0, [], []

        for (wav1, wav2, eeg, label, batch_idx) in dataloader:
            if len(label.shape) == 1:
                label = func.one_hot(label, num_classes=2).float()
            local.aad_model.optim.zero_grad()

            out, all_target, weights = local.aad_model.model(wav1, wav2, eeg, label)

            loss = local.aad_model.loss(out, all_target)
            losses = losses + loss.cpu().detach().numpy()
            cos.append(out.cpu().detach().numpy())

            predict.append(out)
            targets.append(all_target.cpu().detach().numpy())
            if mode == "train":
                loss.backward()
                local.aad_model.optim.step()

        average_loss = losses / len(dataloader)
        output = (predict, targets, average_loss) if mode == "test" else average_loss
        return cos, targets


def cos_analysis(dataloader, local, args) -> np.ndarray:
    tng_loader, tes_loader = data_loader
    cos1, target1 = get_cos(tng_loader, local, args)
    cos2, target2 = get_cos(tes_loader, local, args)
    cos1.extend(cos2)
    target1.extend(target2)
    cos1 = np.array(cos1).reshape(-1, 2)
    target1 = np.array(target1).reshape(-1, 2)

    for i in range(cos1.shape[0]):
        if target1[i, 0] == 0:
            cos1[i, 0], cos1[i, 1] = cos1[i, 1], cos1[i, 0]

    return cos1


# 读取模型等
# 初始化
local, args = init(name="S1", log_path="./result/test", local=DotMap(), args=DotMap())
process_steps = [read_extra_audio, subject_split]
process_steps += [remove_repeated, add_negative_samples]
train_steps = [load, get_data_loader]
data_loader = None
for j in range(len(train_steps)):
    local.logger.info("working process: " + train_steps[j].__name__)
    data_loader, args, local = train_steps[j](data_loader=data_loader, args=args, local=local)

cos = cos_analysis(dataloader=data_loader, args=args, local=local)

plt.hist(cos.flatten(), bins=1000)
plt.title(f"total")
plt.show()

plt.hist(cos[..., 0].flatten(), bins=1000)
plt.title(f"attend")
plt.show()

plt.hist(cos[..., 1].flatten(), bins=1000)
plt.title(f"unattend")
plt.show()

dis = cos[..., 0] - cos[..., 1]
plt.title(f"distance = attend - unattend")
plt.hist(dis.flatten(), bins=1000)
plt.show()