import os
import glob
import time
import shutil
import logging
from importlib import reload
import re
from decimal import Decimal
import numpy as np
import argparse
from multiprocessing import Process
import eutils.util as util
from dotmap import DotMap
from multiprocessing.managers import SharedMemoryManager
import copy

logger = 0
result_document = "./result"

# 参数
args = None
max_procedure = 18
current_flod = 0
need_shared_menory = False
file_name = ""
isCV = False
wait = 120


def share_menory(smm, data):
    shm_d = smm.SharedMemory(size=data.nbytes)
    shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm_d.buf)
    np.copyto(shared_data, data)
    return shm_d


def multiple_process(names, func, multiple, second):
    _log_path = f"{result_document}/{args.label}"
    _log_path += "" if not isCV else f"/flod_{current_flod}"

    if need_shared_menory:
        pass

    process = []
    for name in names:
        _local = DotMap(current_flod=current_flod, core_logger=logger)
        if need_shared_menory:
            pass
        p = Process(target=func, args=(name, _log_path, _local, copy.deepcopy(args)))  # 必须加,号
        p.start()
        time.sleep(wait)
        process.append(p)
        util.monitor(process, multiple, second)

    for p in process:
        p.join()


def __get_last_line(filename):
    try:
        filesize = os.path.getsize(filename)
        if filesize == 0:
            return None
        else:
            with open(filename, 'rb') as fp:  # to use seek from end, must use mode 'rb'
                offset = -2  # initialize offset
                while -offset < filesize:  # offset cannot exceed file size
                    # read # offset chars from eof(represent by number '2')
                    fp.seek(offset, 2)
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:  # if contains at least 2 lines
                        # then last line is totally included
                        return lines[-1]
                    else:
                        offset *= 2  # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return lines[-1]
    except FileNotFoundError:
        logger.error(filename + ' not found!')
        return ""


def search_all_files_return_by_time_reversed(path, reverse=True):
    return sorted(glob.glob(os.path.join(path, '*')),
                  key=lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(x))),
                  reverse=reverse)


def MonitorParameters():
    while True:
        file_list = search_all_files_return_by_time_reversed("./parameters")
        if len(file_list) > 0:
            if os.path.exists("./parameters.py"):
                os.remove("./parameters.py")
            global file_name
            file_name = file_list[0][:-3]
            os.rename(file_list[0], "./parameters.py")

            break
        time.sleep(60)
    return


def init():
    import parameters
    from eutils.update_split_utils import cv_divide
    reload(parameters)
    global args
    global isCV
    args = copy.deepcopy(parameters.args)
    isCV = False
    for func in args.proc_steps:
        if cv_divide.__name__ == func.__name__:
            isCV = True

    if os.path.exists(f"{result_document}/{args.label}"):
        args.label = f"{args.label}_{time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())}"
    save_path = f"{result_document}/{args.label}"

    os.mkdir(save_path)
    shutil.copy("./parameters.py", f"{save_path}/{args.label}.py")

    reload(logging)
    global logger
    # 第一步，创建一个logger
    logger = logging.getLogger("core_logger")
    logger.setLevel(logging.INFO)

    # 第二步，创建一个handler，用于写入日志文件
    logfile = f"{save_path}/logger.txt"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 第四步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"model label: {args.label} {file_name}")


def output_result():
    output = "result: \n"
    totle = 0
    _log_path = f"{result_document}/{args.label}"
    _log_path += "" if not isCV else f"/flod_{current_flod}"
    for i in range(len(args.names)):
        filename = _log_path + "/Train_" + args.names[i] + ".log"
        _str = __get_last_line(filename).decode()
        output = output + _str
        # 匹配小数部分，然后平均
        _str = re.search("(-?\d+)(\.\d+)?", _str).group()
        _str = Decimal(_str).quantize(Decimal('0.0000'))
        totle = totle + float(_str)
    average = totle / len(args.names)
    logger.info(output)
    logger.info(average)
    return average


def train():
    from main import main as model_training
    logger.info("train start!")
    multiple_process(args.names, model_training, max_procedure, 60)
    logger.info("train finish!")


def grid_search(filename, pattern, parameter_range):
    best_perameter = None
    best_average = 0

    for i in range(len(parameter_range)):
        file = open(filename, "r", encoding=("utf-8"))
        string = file.read()
        string = re.sub(pattern, parameter_range[i], string)
        file.close()

        file = open("parameters.py", "w", encoding=("utf-8"))
        file.write(string)
        file.close()

        init()
        logger.info("parameter: " + parameter_range[i])
        train()

        average = output_result()
        if average > best_average:
            best_average = average
            best_perameter = parameter_range[i]

    logger.info(f"best average: {best_average}")
    logger.info(f"best perameter: {best_perameter}")


def loopMonitor():
    global current_flod
    while True:
        MonitorParameters()
        init()
        if not isCV or args.split_meta.curr_flod is not None:
            current_flod = args.split_meta.curr_flod
            train()
            output_result()
        else:
            for i in range(args.split_meta.cv_flod):
                current_flod = i
                args.split_meta.curr_flod = i
                train()
                output_result()


if __name__ == "__main__":
    # 暂时先合并三个文件，后续再扩展
    parser = argparse.ArgumentParser(description="Boot the train")
    parser.add_argument("-m", "--mode", required=True, choices=["train", "grid_search"],
                        help="mode: train, grid_search")
    parser.add_argument("-f", "--file", type=str, default="./parameters/dot_baseline_delay.py",
                        help="the orgin file need to be opti")
    parser.add_argument("-p", "--pattern", type=str, default="lr=(-?\d+)(\.\d+)?",
                        help="the pattern need to be replaced")
    parser.add_argument("-n", "--number", type=float, nargs="+",
                        help="range: min, max and step; p.s.: 5 decimal places will be retained")
    parser.add_argument('--mul', type=int, help="max number of multiprocessdure ")
    _args = parser.parse_args()
    max_procedure = _args.mul

    # 正则匹配式
    pattern = "TD = (-?\d+)"

    if _args.mode == "grid_search":
        # temp_list = np.arange(args.number[0], args.number[1], args.number[2])
        # parameter_range = ["ken_length = " + str(('%.5f' % temp_list[i])) for i in range(temp_list.shape[0])]
        # 范围
        temp_list = [i for i in range(16)]
        parameter_range = ["TD = " + str(temp_list[i]) for i in range(len(temp_list))]
        grid_search(_args.file, pattern, parameter_range)
    elif _args.mode == "train":
        loopMonitor()
