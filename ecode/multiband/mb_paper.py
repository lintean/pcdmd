from mne.time_frequency.csd import csd_array_multitaper
import numpy as np
from matplotlib import pyplot as plt
from dotmap import DotMap
import eutils.util as util
import pandas as pd
import scipy.io as io
import sys


def data_process(train_window, data, window_metadata):
    input_data = []
    for i in range(train_window.shape[0]):
        start = train_window[i, window_metadata.start]
        end = train_window[i, window_metadata.end]
        input_data.append(data[start:end, :])

    input_data = np.stack(input_data, axis=0)
    input_data = input_data.transpose((0, 2, 1))
    # print(input_data.shape)
    # print(data.shape)
    return input_data


def coherence_cul(name="S1", log_path="./result/mb_paper"):
    data_document = "./data/multiband"
    log_path = util.makePath(log_path)
    data_meta = util.read_json(data_document + "/metadata.json")

    file = np.load(data_document + "/" + name + ".npz", allow_pickle=True)
    train_window = file["train_window"]
    data = file["data"]
    # pd.DataFrame(data).to_csv(log_path + "/data.csv", header=None, index=None)
    window_metadata = DotMap(file["window_metadata"].item())
    del file

    mb_low = [1, 3, 7, 13, 1, 1, 1, 1]
    mb_high = [3, 7, 15, 30, 7, 7, 7, 7]
    audio_channel = data_meta.audio_band * data_meta.audio_channel_per_band
    env_att = data[:, 0:audio_channel]
    env_ign = data[:, audio_channel:audio_channel * 2]
    start = 2 * audio_channel
    coherence = []

    for i in range(data_meta.eeg_band):
        EEG = data[:, start + data_meta.eeg_channel_per_band * i:
                      start + data_meta.eeg_channel_per_band * (i + 1)]
        mb_data = np.concatenate([env_att, env_ign, EEG], axis=1)

        input_data = data_process(train_window=train_window, data=mb_data, window_metadata=window_metadata)
        # shape(n_epochs, n_channels, n_times)
        result = csd_array_multitaper(input_data, sfreq=256, fmin=mb_low[i], fmax=mb_high[i])
        mean = result.mean()
        # mean.plot()
        mean = mean.get_data()

        coherence_temp = np.zeros([2 * audio_channel, data_meta.eeg_channel_per_band])
        # print(coherence_temp.shape)
        for k in range(coherence_temp.shape[0]):
            for l in range(coherence_temp.shape[1]):
                eeg_index = 2 * audio_channel + l
                coherence_temp[k, l] = abs(mean[k, eeg_index]) / (
                            abs(mean[k, k]) * abs(mean[eeg_index, eeg_index])) ** 0.5
                # print(coherence_temp)

        coherence.append(coherence_temp)

    coherence = np.concatenate(coherence, axis=1)
    # ave = np.array([np.mean(coherence[:audio_channel, :]), np.mean(coherence[audio_channel:, :])])
    # io.savemat(log_path + '/Ave' + name + '.mat', {'Ave': ave})
    pd.DataFrame(coherence).to_csv(log_path + '/Coherence' + name + '.csv', header=None, index=None)


