#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   audio.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/24 21:00   lintean      1.0         None
'''
import math
import numpy as np


def freqtoaud(fmin, fmax):
    freq = [fmin, fmax]
    aud = 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)
    return aud


def audtofreq(aud):
    freq = (1 / 0.00437) * np.sign(aud) * (np.exp(np.abs(aud) / 9.2645) - 1)
    return freq


def audspacebw(fmin=150, fmax=8000, bw=3):
    audlimits = freqtoaud(fmin, fmax)
    audrange = audlimits[1] - audlimits[0]
    n = int(np.floor(audrange / bw))
    remainder = audrange - n * bw
    audpoints = audlimits[0] + np.array([i for i in range(n + 1)]) * bw + remainder / 2
    n = n + 1
    y = audtofreq(audpoints)
    return y


def audfiltbw(fc):
    return 24.7 + fc / 9.265

def factorial(n):
    N = n
    thres = 171
    N = min(N, thres)
    m = max(1, np.max(N))
    Fa = np.product([1, 1] + [i for i in range(2, m + 1)])
    return Fa


def gammatonefir(fc,fs,betamul):
    nchannels = len(fc)
    ourbeta = betamul * audfiltbw(fc)
    n = 5000
    b = []
    offset = []

    for ii in range(nchannels):
        delay = 3 / (2 * np.pi * ourbeta[ii])
        scalconst = 2 * (2 * np.pi * ourbeta[ii]) ** 4 / factorial(4 - 1) / fs
        nfirst = np.ceil(fs * delay)
        if nfirst > n / 2:
            print("error")
            exit(1)
        nlast = np.floor(n / 2)
        tempf = np.array([i for i in range(int(nfirst))])
        templ = np.array([i for i in range(int(nlast))])
        t = np.concatenate([tempf / fs - nfirst / fs + delay, templ / fs + delay])[None, ...]
        bwork = scalconst * np.power(t, (4 - 1)) * np.cos(2 * np.pi * fc[ii] * t) * np.exp(-2 * np.pi * ourbeta[ii] * t)
        b.append(bwork)
        offset.append(-nfirst)

    return b, offset



if __name__ == "__main__":
    from gammatone.filters import make_erb_filters, erb_space, erb_filterbank
    fc = audspacebw()
    g = gammatonefir(fc, fs=20000, betamul=3)
    print(g)
