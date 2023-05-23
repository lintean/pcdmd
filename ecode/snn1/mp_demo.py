import numpy as np
from torch.multiprocessing import Process, set_start_method, Queue
import time
from ecode.GraduationDesign.Demo import demo
from ecfg import project_root_path
import eutils.util as util
import argparse
import pandas as pd
try:
     set_start_method('spawn')
except RuntimeError:
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Boot the train")
    parser.add_argument('--mul', type=int, help="max number of multiprocessdure ")
    parser.add_argument('--start', type=int, help="start index")
    parser.add_argument('--end', type=int, help="end index")
    args = parser.parse_args()
    max_procedure = args.mul
    start = args.start
    end = args.end


    process = []
    q = Queue()
    for name in range(start, end):
        p = Process(target=demo, args=(str(name), True, project_root_path + "/result/snn", q, ))  # 必须加,号
        p.start()
        time.sleep(30)
        process.append(p)
        util.monitor(process, max_procedure, 60)

    for p in process:
        p.join()

    result = []
    for i in range(len(process)):
        result.append(q.get())
    result = np.stack(result, axis=0)
    pd.DataFrame(result).to_csv(project_root_path + "/result/snn/result.csv", header=["name", "best_acc", "best_acc_epoch", "snn_energy", "cnn_energy", "total_spike", "total_all", "total_spike/total_all"], index=False)

