from scipy.signal import hilbert
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torch.autograd import Function
from torch.autograd import Variable
import math
import random
import sys
from parameters import *
import seaborn as sns
from eutils.subFunCSP import learnCSP as GitCSP

def main(data_document="./data/split_dot_test"):

    for j in range(1, 19):
        name = "S" + str(j)
        print("train start!")
        data = np.load("./" + data_document + "/CNN1_" + name + ".npy", allow_pickle=True)
        data = data[0]

        output = []
        for i in range(data.shape[0]):
            d = dict()
            d["windowIndex"] = data[i]["index"]
            temp = data[i]["direction"]
            d["attend"] = "attend=A" if temp - 1==0 else "attend=B"
            output.append(d)

        df = pd.DataFrame(output, columns=["windowIndex", "attend"])
        df.to_csv("./log/attend/" + name + ".csv")
        # print(df)

if __name__ == "__main__":
    main()