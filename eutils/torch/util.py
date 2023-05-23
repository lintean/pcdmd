#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   util.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/23 17:25   lintean      1.0         None
'''
from typing import Tuple, Any

from dotmap import DotMap
import torch

from .container import AADModel
from ..util import makePath


def save_model(aad_model: AADModel, seed: int, local: DotMap) -> None:
    """
    保存模型
    @param seed:
    @param local:
    @param aad_model:
    @return:
    """
    torch.save(
        {
            'epoch': local.epoch,
            'state_dict': aad_model.model.state_dict(),
            'optimizer': aad_model.optim.state_dict(),
            'seed': seed
        },
        f"{makePath(local.log_path)}/model-{local.name}.pth.tar"
    )


def load_model(file_name: str, aad_model: AADModel, **kwargs) -> tuple[AADModel, int]:
    """
    加载模型
    @param file_name:
    @param aad_model:
    @return:
    """
    model_CKPT = torch.load(f"{file_name}", **kwargs)
    aad_model.model.load_state_dict(model_CKPT['state_dict'])
    aad_model.optim.load_state_dict(model_CKPT['optimizer'])

    if 'seed' in model_CKPT:
        return aad_model, int(model_CKPT['seed'])
    else:
        return aad_model
