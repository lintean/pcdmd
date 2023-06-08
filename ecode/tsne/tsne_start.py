import pandas as pd
import numpy as np
from dotmap import DotMap
from ecfg import project_root_path
# 主要用到了openTSNE的库
from openTSNE import TSNE
import eutils.tsne as TSNEUtil
import os
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from tqdm import tqdm
from eutils.util import makePath


def tsne_run(model_type, flod, name, module):
    basedir = f"{project_root_path}/result/BSApic/{model_type}_KUL_1/{flod}/{name}"
    data_path = []

    for root, dirs, files in os.walk(basedir):
        for file in files:
            path = os.path.join(root, file)
            if fnmatch(path, f"*_{module}_*.csv"):
                data_path.append(path)
    data_path.sort()

    x = []
    y = []
    hash = {1:0, 2:0, 3:1, 4:1}
    for i in range(len(data_path)):
        data = pd.read_csv(data_path[i], header=None)
        x.append(data.to_numpy().flatten())
        y.append(hash[int(data_path[i][-5])])

    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=False
    )
    x = np.array(x)
    y = np.array(y)

    embedding = tsne.fit(x)
    path = f"{project_root_path}/result/BSApic/tsne/{flod}/{name}"
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    TSNEUtil.plot(embedding, y, colors=TSNEUtil.EEG_COLOR, ax=ax)
    fig.savefig(f"{makePath(path)}/{model_type}_{module}.png", format='png', transparent=False)
    # plt.show()
    plt.close(fig)
    pass


def model_tsne(model_type, flod, name):
    modules = ()
    if model_type == "RSNN":
        modules = ("encoder", "rsnn")
    elif model_type == "SSA":
        modules = ("encoder", "rsnn", "sa")

    for module in modules:
        tsne_run(model_type, flod, name, module)


def main(local: DotMap = DotMap(), args: DotMap = DotMap()):
    model_type = "SSA"
    flods = [f"flod_{i}" for i in range(5)]
    names = [f"S{i + 1}" for i in range(16)]

    with tqdm(total=len(flods) * len(names)) as pbar:
        for flod in flods:
            for name in names:
                model_tsne(model_type=model_type, flod=flod, name=name)
                pbar.update(1)

if __name__ == "__main__":
    main()
    # from sklearn import datasets
    # iris = datasets.load_iris()
    # x, y = iris["data"], iris["target"]
    # embedding = TSNE().fit(x)
    # TSNEUtil.plot(embedding, y, colors=TSNEUtil.MOUSE_10X_COLORS)