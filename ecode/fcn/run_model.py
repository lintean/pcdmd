from pathlib import Path
from multiprocessing import Process
from ecode.fcn.Code import main
import eutils.util as util

if __name__ == "__main__":
    multiple = 4
    process = []
    delays = [i for i in range(3, 21)]
    names = ['S' + str(i+1) for i in range(0, 18)]
    for delay in delays:
        for name in names:
            path = "./result/fcn/delay" + str(delay)
            file = Path(path + '/CorrTest' + name + '.mat')
            if not file.exists():
                p = Process(target=main, args=(name, delay, path,))  # 必须加,号
                p.start()
                process.append(p)
                util.monitor(process, multiple, 60)


