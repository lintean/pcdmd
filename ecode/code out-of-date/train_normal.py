import numpy as np
import torch
import math
import random
import sys
import parameters
import eutils.util as util
from dotmap import DotMap
from tqdm import trange
import time
import scipy.io as io
from importlib import reload


def get_logger(name, log_path):
    import logging
    reload(logging)

    # 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 第二步，创建一个handler，用于写入日志文件
    logfile = log_path + "/Train_" + name + ".log"
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


def modeling(data, windows, mode, window_metadata, need_predict):
    args = parameters.args
    losses = 0
    predict = []
    targets = []
    env_as = []
    env_uns = []
    data = torch.tensor(data, dtype=torch.float32).to(parameters.device) if not torch.is_tensor(data) and not args.need_pretrain else data

    for turn in range(math.floor(windows.shape[0] / args.batch_size)):
        parameters.optimzer.zero_grad()
        batch_data = []
        all_target = []
        env_a = []
        env_un = []
        for k in range(args.batch_size):
            start = windows[turn * args.batch_size + k, window_metadata.start]
            end = windows[turn * args.batch_size + k, window_metadata.end]

            batch_data.append(data[start:end, :].transpose(1,0))
            all_target.append(windows[turn * args.batch_size + k, window_metadata.target] - 1)
            env_a.append(data[start, :args.audio_channel])
            env_un.append(data[start, args.audio_channel:args.audio_channel * 2])

        if torch.is_tensor(batch_data[0]):
            batch_data = torch.stack(batch_data, dim=0)[:, None, :, :]
        else:
            batch_data = np.stack(batch_data, axis=0)[:, None, :, :]
            batch_data = torch.tensor(batch_data, dtype=torch.float32).to(parameters.device)
        all_target = torch.tensor(np.array(all_target), dtype=torch.long).to(parameters.device)

        out = parameters.myNet(batch_data)
        loss = parameters.loss_func(out, all_target)
        losses = losses + loss.cpu().detach().numpy()

        predict.append(out)
        targets.append(all_target)
        env_as.append(env_a)
        env_uns.append(env_un)
        if mode == "train":
            loss.backward()
            parameters.optimzer.step()

    average_loss = losses / (math.floor(windows.shape[0] / args.batch_size))
    output = [predict, targets, average_loss, env_as, env_uns] if need_predict else average_loss
    return output


def trainEpoch(data, train_window, test_window, window_metadata, local):
    min_loss = 100
    early_stop_number = 0
    train_window_backup = train_window.copy()

    for epoch in range(parameters.args.max_epoch):
        # 打乱非测试数据集并划分训练集和验证集
        train_window, vali_window = util.vali_split(train_window_backup, parameters.args)
        np.random.shuffle(train_window)

        # 训练和验证
        loss_train = modeling(data, train_window, "train", window_metadata, False)
        loss = modeling(data, vali_window, "vali", window_metadata, False)
        loss2 = modeling(data, test_window, "test", window_metadata, False)

        # 学习率衰减
        # scheduler.step()
        parameters.scheduler.step(0.1)

        if loss > min_loss:
            early_stop_number = early_stop_number + 1
        else:
            early_stop_number = 0
            min_loss = loss

        info = str(epoch) + " " + str(loss_train) + " " + str(loss) + " " + str(loss2)
        local.logger.info(info + " early_stop_number: " + str(early_stop_number))

        if parameters.args.early_patience > 0 and epoch > parameters.args.min_epoch and early_stop_number >= parameters.args.early_patience:
            break


def testEpoch(data, test_window, window_metadata, local):

    if parameters.args.is_regression:
        local.logger.info("reg test")
        predict, targets, env_a, env_u, ave_loss = modeling(data, test_window, "test", window_metadata, True)
        corr_results = util.evaluation(predict, env_a, env_u, parameters.args)
        io.savemat(local.log_path + "/" + local.name + '.mat', {'corrResults' + local.name: corr_results})
    else:
        total_t_num = 0
        total_f_num = 0
        for num in range(10):
            t_num = 0
            f_num = 0
            predict, targets, env_a, env_u, ave_loss = modeling(data, test_window, "test", window_metadata, True)

            for turn in range(len(predict)):
                out = predict[turn]
                allTarget = targets[turn]

                for i in range(parameters.args.batch_size):
                    result = out[i][None, :]
                    lossL = parameters.loss_func(result, torch.tensor([0]).to(parameters.device)).cpu().detach().numpy()
                    lossR = parameters.loss_func(result, torch.tensor([1]).to(parameters.device)).cpu().detach().numpy()
                    if (lossL < lossR) == (allTarget[i].cpu().detach().numpy() == 0):
                        t_num = t_num + 1
                    else:
                        f_num = f_num + 1

            local.logger.info(str(t_num) + " " + str(f_num))
            total_t_num = total_t_num + t_num
            total_f_num = total_f_num + f_num
        local.logger.info("\n" + str(total_t_num / (total_t_num + total_f_num)))


def main(name="S1", log_path="./result/test"):
    reload(parameters)

    logger = get_logger(name, log_path)

    global myNet
    global loss_func
    parameters.myNet = parameters.myNet.to(parameters.device)
    parameters.loss_func = parameters.loss_func.to(parameters.device)
    logger.info(id(parameters.myNet))
    logger.info(parameters.device)

    data = None
    train_window = None
    test_window = None
    window_metadata = None
    local = DotMap()
    local.name = name
    local.log_path = log_path
    local.logger = logger

    # 跨进程读取数据并预训练
    if parameters.args.need_pretrain:
        logger.info("pretrain start!")
        # trainEpoch(data, train_window, test_window)

    # 读取数据、同被试训练
    if parameters.args.need_train:
        # # 如果有预训练则降低学习率
        # if args.need_pretrain:
        #     for p in optimzer.param_groups:
        #         p['lr'] *= 0.1

        logger.info("train start!")
        file = np.load(parameters.args.data_document + "/" + name + ".npz", allow_pickle=True)
        train_window = file["train_window"]
        test_window = file["train_window"] if parameters.args.isALLTrain and parameters.args.need_pretrain and not parameters.args.need_train else file[
            "test_window"]
        data = file["data"]
        window_metadata = DotMap(file["window_metadata"].item())
        del file

        trainEpoch(data, train_window, test_window, window_metadata, local)
        logger.info("train finish!")

    # 测试
    logger.info("test start!")
    testEpoch(data, test_window, window_metadata, local)

if __name__=="__main__":
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1])
    else:
        main()

