from pathlib import Path
from multiprocessing import Process
from ecode.multiband.mb_paper import coherence_cul
import eutils.util as util

if __name__ == "__main__":
    multiple = 2
    process = []
    names = ['S' + str(i+1) for i in range(0, 18)]
    for name in names:
        path = "./result/mb_paper"
        # file = Path(path + '/CorrTrain' + name + '.mat')
        # if not file.exists():
        p = Process(target=coherence_cul, args=(name, path,))  # 必须加,号
        p.start()
        process.append(p)
        util.monitor(process, multiple, 60)


