import pandas as pd
import eutils.util as util
import numpy as np
from pathlib import Path
from multiprocessing import Process
import scipy.io as io
import math

def process_mat(document_path, log_path, delay):
    names = ['S' + str(i+1) for i in range(0, 18)]
    results = []
    for name in names:
        log_path = util.makePath(log_path)
        path = document_path + '/CorrTest' + name + '.mat'
        file = Path(path)
        if not file.exists():
            results.append(np.zeros([7]))
        else:
            result = io.loadmat(path)['CorrTest' + name]
            if len(results) == 0: results.append(result[6])
            results.append(result[4])
        # if name == "S9":
        #     print(result)
    results = np.stack(results, axis=0)
    pd.DataFrame(results).to_csv(log_path + "/results" + str(delay) + ".csv", header=None, index=None)

def process_csv(document_path, log_path):
    names = ['S' + str(i + 1) for i in range(0, 18)]
    windows = [60, 30, 10, 5, 2, 1, 0.5]
    delays = [i for i in range(3, 21)]
    aves = []
    for name in names:

        results = []
        for delay in delays:
            log_path = util.makePath(log_path)
            path = document_path + '/results' + str(delay) + '.csv'
            file = Path(path)
            result = np.array(pd.read_csv(path, header=None, index_col=None))
            name_num = int(name[1:])
            results.append(result[name_num,:])
        results = np.stack(results, axis=0)
        aves.append(results)

    aves = np.stack(aves, axis=0)
    # delays = np.repeat(np.array(delays)[None,:,None], len(windows), axis=2)
    # aves = np.concatenate([delays, aves], axis=0)
    # aves = np.ascontiguousarray(aves, dtype=np.float32)

    for i in range(aves.shape[2]):
        pd.DataFrame(aves[:,:,i]).to_csv(log_path + "/window" + str(math.floor(windows[i])) + ".csv", header=delays)


if __name__ == "__main__":
    multiple = 2
    process = []
    delays = [i for i in range(3, 21)]
    names = ['S' + str(i + 1) for i in range(0, 18)]

    process_csv("./result/fcn/results", "./result/fcn/results2")
    # for name in names:
    # # for delay in delays:
    #     path = "./result/fcn/results"
    #     log_path = "./result/fcn/results2"
    #     p = Process(target=process_csv, args=(path, log_path, name))  # 必须加,号
    #     p.start()
    #     p.join()
    #     process.append(p)
    #     # util.monitor(process, multiple, 60)