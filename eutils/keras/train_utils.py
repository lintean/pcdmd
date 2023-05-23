#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_utils.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/17 10:19   lintean      1.0         None
'''
from dotmap import DotMap
from CNN_split import split_data
from .container import DataGenerator


def get_data_loader(local: DotMap = DotMap(), args: DotMap = DotMap(), **kwargs):
    """
    获取数据集
    @param args:
    @param local:
    @return:
    """
    data, train_window, test_window, window_metadata = split_data(local.name, log_path=local.log_path, args=args, local=local)
    window_metadata = DotMap(window_metadata)

    params = {'batch_size': args.batch_size,
              'wave_chan': args.audio_channel,
              'eeg_chan': args.eeg_channel,
              'eeg_len': args.window_length,
              'wave_len': args.window_length,
              'shuffle': True}

    wav1, wav2, eeg = data[0].transpose(-1, -2), data[1].transpose(-1, -2), data[2].transpose(-1, -2)
    tng_gen = DataGenerator(eeg=eeg, wav1=wav1, wav2=wav2, window=train_window, window_meta=window_metadata, **params)
    tes_gen = DataGenerator(eeg=eeg, wav1=wav1, wav2=wav2, window=test_window, window_meta=window_metadata, **params)

    return (tng_gen, tes_gen), args, local


def single_model(data_loader: tuple, local: DotMap = DotMap(), args: DotMap = DotMap()):
    tng_gen, tes_gen = data_loader
    local.logger.info("train start!")
    local.model.compile(
        optimizer=local.optimizer,
        loss=local.loss_func,
        metrics=['accuracy'],
    )
    local.model.fit_generator(
        generator=tng_gen,
        validation_data=tes_gen,
        # use_multiprocessing=True,
        # workers=6,
        epochs=local.max_epoch,
        callbacks=local.callbacks
    )

    local.logger.info("train finish!")
    # 测试最后一个模型
    local.logger.info("test start!")
    loss, acc = local.model.evaluate_generator(generator=tes_gen)
    local.logger.info(f"loss: {loss}, acc: {acc}")

    return None, args, local