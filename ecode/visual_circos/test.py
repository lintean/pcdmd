#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/18 15:59   lintean      1.0         None
'''
import numpy as np
import os
from eutils.util import makePath

colorsL = ['Fp', 'AF', 'F', 'FT', 'FC', 'T', 'C', 'TP', 'CP', 'P', 'PO', 'O', 'I']
colorsD = dict(
    {'Fp': 1, 'AF': 1, 'F': 1, 'FT': 2, 'FC': 2, 'T': 2, 'C': 2, 'TP': 2, 'CP': 2, 'P': 3, 'PO': 3, 'O': 3, 'I': 3})
labels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5',
          'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
          'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6',
          'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
ordered_lables = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
                  'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P9',
                  'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1',
                  'Oz', 'O2', 'Iz']


def top_n_largest(data: np.ndarray, n: int = 100):
    def data_cmp(k1, k2):
        return 0 if k1[0] == k2[0] else 1 if k1[0] > k2[0] else -1

    subject, flod, channel, ch1, ch2 = data.shape
    to_list = []
    for s in range(subject):
        for f in range(flod):
            for c in range(channel):
                for i in range(ch1):
                    for j in range(ch2):
                        if i != j:
                            to_list.append((data[s, f, c, i, j], s, f, c, i, j))

    return sorted(to_list, reverse=True)[:n]


def rm_duplicates(data: list, n: int = 10):
    repeat = dict()
    result = []
    for i in range(len(data)):
        if n == 0: break
        chr1 = labels[data[i][-2]]
        chr2 = labels[data[i][-1]]
        chr1_index = ordered_lables.index(chr1)
        chr2_index = ordered_lables.index(chr2)
        if chr1_index > chr2_index:
            chr1, chr2 = chr2, chr1
        if chr1 in repeat and repeat[chr1] == chr2: continue
        repeat[chr1] = chr2
        n = n - 1
        result.append((chr1, chr2, data[i][0]))
    return result


def output(data, save_path):
    txt = ""
    maxm, minm = data[0][2], data[-1][2]

    for i in range(len(data)):
        chr1 = data[i][0]
        chr2 = data[i][1]
        chr1_type = chr1[:-2] if chr1[-2].isdigit() else chr1[:-1]
        chr2_type = chr2[:-2] if chr2[-2].isdigit() else chr2[:-1]
        # color = 'vvdred' if colorsD[chr1_type] != colorsD[chr2_type] else f'rdylbu-3-div-{colorsD[chr1_type]}'
        transparency = (data[i][2] - minm) / (maxm - minm)
        color = f'201,0,0,{transparency}'
        txt = txt + f'{chr1} 0 3 {chr2} 0 3 color={color}\n'

    with open(save_path, 'w') as file:
        file.write(txt)


file_names = [0, 623, 1246, 1868, 2490]
circos_path = f'E:\\school\\origin\\EEG\\peiwen\\ecode\\visual_circos'
save_path = f'E:\\open\\circos-0.69-9\\test\\data\\data.txt'
subject_data = []

for s in range(1, 17):
    subject_data_temp = []
    for file_name in file_names:
        data_path = f'D:\\eegdata\\visual\\KUL_GCN\\data\\S{s}\\V{file_name}.npz'
        temp = np.load(data_path)
        subject_data_temp.append(temp['w_graph'])
        pass
    subject_data.append(np.concatenate(subject_data_temp, axis=0))


circos_save_path = f"{circos_path}\\pic\\total"
makePath(circos_save_path)
for flod in range(6):
    data = np.stack(subject_data, axis=0)
    data = np.average(data, axis=0)[None, ...]
    data = data[:, flod:flod + 1, :, :, :] if flod < 5 else np.average(data, axis=1)[:, None, ...]
    data = top_n_largest(data, 100)
    data = rm_duplicates(data, 10)
    output(data, save_path)
    os.system(f"perl E:\\open\\circos-0.69-9\\bin\\circos -conf E:\\open\\circos-0.69-9\\test\\circos.conf -outputdir {circos_save_path} -outputfile {flod + 1}.png")

# makePath(f'{circos_path}\\pic')
# for subject in range(16):
#     circos_save_path = f"{circos_path}\\pic\\S{subject + 1}"
#     makePath(circos_save_path)
#     for flod in range(6):
#         temp = subject_data[subject][None, ...]
#         temp = temp[:, flod:flod + 1, :, :, :] if flod < 5 else np.average(temp, axis=1)[:, None, ...]
#         temp = top_n_largest(temp, 100)
#         temp = rm_duplicates(temp, 10)
#         output(temp, save_path)
#         os.system(
#             f"perl E:\\open\\circos-0.69-9\\bin\\circos -conf E:\\open\\circos-0.69-9\\test\\circos.conf -outputdir {circos_save_path} -outputfile {flod + 1}.png")
