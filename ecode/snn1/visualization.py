#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visualization.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/3 19:48   lintean      1.0         None
'''
import math
import eutils.util as util
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ecfg import project_root_path
import re


def __get_figsize(x, y):
    min = 1
    max = 30
    scale = 3
    figsize = None
    if min <= x/scale <= max and min <= y/scale <= max:
        figsize = [math.ceil(x/scale), math.ceil(y/10)]
    if x/scale > max or y/scale > max:
        max_size = x if x>y else y
        figsize = [x/max_size * max, y/max_size * max]
    if x/scale < min or y/scale < min:
        min_size = x if x<y else y
        figsize = [x/min_size * min, y/min_size * min]
    return figsize


def heatmap(_, log_path=None):
    a = _
    figsize = [10, 10]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.)
    canvas = pd.DataFrame(np.round(a, 2))
    heatmap = sns.heatmap(canvas, xticklabels=False, yticklabels=False, square=True, cbar=True, vmax=1, vmin=0)

    # plt.show()
    fig.savefig(log_path + ".png", transparent=True)
    plt.close(fig)


save_path = project_root_path + "\\result\\snn_visual"
visual_path = project_root_path + "\\parameters_backup\\temp\\20210605\\visual\\visual.csv"
visual_file = pd.read_csv(visual_path, header=None, index_col=None)
visual_file = visual_file.to_numpy()
for i in range(visual_file.shape[0]):
    person_label = visual_file[i, 1]
    index_label = visual_file[i, 2]
    epoch_label = visual_file[i, 3]
    document_path = project_root_path + "\\parameters_backup\\temp\\20210605\\visual\\snn\\" + person_label + "\\picture\\" + index_label
    file_pattern = epoch_label + ".*.csv"
    files = util.get_sub_files(document_path, find_dir=False)
    for file_name in files:
        if re.match(file_pattern, file_name):
            file = pd.read_csv(document_path + "\\" + file_name, header=None, index_col=None)
            file = file.to_numpy()
            if np.sum(file) != 0:
                file = util.normalization(file)
            heatmap(file, log_path=util.makePath(save_path) + "\\" + person_label + " " + index_label + " " + file_name[:-4])


