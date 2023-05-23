import sys
import importlib
import device_to_use
from eutils.util import makePath, get_gpu_with_max_memory
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
    if not args:
        module = importlib.import_module(f"parameters")
        args = module.args

    logger = get_logger(name, log_path)

    local.name = name
    local.name_index = int(local.name[1:]) - 1
    local.log_path = log_path
    local.logger = logger

    local.logger.info(f'data path: {args.data_document_path}')
    local.logger.info(f'window length: {args.window_length}')
    local.logger.info(f'overlap: {args.overlap}')

    args.gpu_list = device_to_use.gpu_list
    aad_model = None
    if "torch" in args.train_steps[0].__module__:
        module = importlib.import_module(f"{args.model_path}")
        aad_model = module.get_model(args)
        args.dev = aad_model.dev
        local.logger.info(aad_model.dev)
        local.aad_model = aad_model

    if "keras" in args.train_steps[0].__module__:
        pass

    local.logger.info(id(aad_model.model))
    return local, args


def main(name: str = "S1", log_path: str = "./result/test", local: DotMap = DotMap(), args: DotMap = DotMap()):
    local, args = init(name, log_path, local, args)

    # 设定流程：
    # 1.读取数据生成dataloader
    # 2.训练并测试模型
    assert "train_steps" in args, f"cannot find train_steps in args"
    train_steps = args.train_steps

    data_loader = None
    for j in range(len(train_steps)):
        local.logger.info("working process: " + train_steps[j].__name__)
        data_loader, args, local = train_steps[j](data_loader=data_loader, args=args, local=local)


if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1])
    else:
        main()
