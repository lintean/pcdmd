import pandas as pd
import eutils.util as util
import numpy as np
from pathlib import Path
from multiprocessing import Process

def figure_three(name, document_path, log_path):
    data_document = "./data/multiband"
    log_path = util.makePath(log_path)
    data_meta = util.read_json(data_document + "/metadata.json")
    coherence = pd.read_csv(document_path + "/Coherence" + name + ".csv", header=None, index_col=None)
    coherence = coherence.to_numpy()
    audio_channel = data_meta.audio_band * data_meta.audio_channel_per_band

    ave_coh = []
    for i in range(data_meta.eeg_band):
        temp = coherence[:, i * data_meta.eeg_channel_per_band:(i+1) * data_meta.eeg_channel_per_band]
        temp = np.mean(temp, axis=1)
        temp = temp[:audio_channel] - temp[audio_channel:]
        ave_coh.append(temp)

    ave_coh = np.stack(ave_coh, axis=1)
    ave_coh = ave_coh.transpose((1,0))
    # Z-score
    ave_coh = (ave_coh - np.mean(ave_coh)) / np.std(ave_coh)
    pd.DataFrame(ave_coh).to_csv(log_path + '/Coherence' + name + '.csv', header=None, index=None)

def mean_upper(coherence, mean):
    temp = []
    for i in range(coherence.shape[0]):
        for j in range(coherence.shape[1]):
            if coherence[i,j] > mean:
                temp.append(coherence[i,j])
    temp = np.array(temp)
    return np.mean(temp)

def figure_five(name, document_path, log_path):
    data_document = "./data/multiband"
    log_path = util.makePath(log_path)
    data_meta = util.read_json(data_document + "/metadata.json")
    coherence = pd.read_csv(document_path + "/Coherence" + name + ".csv", header=None, index_col=None)
    coherence = coherence.to_numpy()
    audio_channel = data_meta.audio_band * data_meta.audio_channel_per_band

    mean = np.mean(coherence)
    for i in range(6):
        mean = mean_upper(coherence, mean)

    count = [0, 0]
    for i in range(coherence.shape[0]):
        for j in range(coherence.shape[1]):
            if coherence[i,j] > mean:
                count[int(i / 10)] += 1

    print(count[0], end="\t")
    print(count[1])

if __name__ == "__main__":
    multiple = 2
    process = []
    names = ['S' + str(i+1) for i in range(0, 18)]
    for name in names:
        path = "./result/mb_paper"
        # file = Path(path + '/CorrTrain' + name + '.mat')
        # if not file.exists():
        log_path = path + "/figure5"
        p = Process(target=figure_five, args=(name, path, log_path,))  # 必须加,号
        p.start()
        p.join()
        process.append(p)
        # util.monitor(process, multiple, 3)
