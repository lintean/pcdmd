import numpy as np
import pandas as pd
import math
import random
import sys


def main():
    temp = None
    for i in range(1, 17):
        name = "S" + str(i)
        data = np.load("./data/ADN2_norF_KUL_all16_band5/CNN1_" + name + ".npy", allow_pickle=True)
        temp = data[0][np.newaxis, :] if temp is None \
            else np.concatenate((temp, data[0][np.newaxis, :]), axis=1)
        print(data[0].shape)
    print(temp.shape)

if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1])
    else:
        main()