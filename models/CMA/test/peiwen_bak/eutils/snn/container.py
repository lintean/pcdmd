#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   container.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/5/25 10:10   lintean      1.0         None
'''

from dataclasses import dataclass, field
from typing import Any, List
import numpy as np
from enum import auto, Enum
import math

@dataclass
class SPower:
    energy: float = field(default=0.0)
    mac: float = field(default=0)
    spikes: float = field(default=0)
    max_spikes: float = field(default=0)
    params: float = field(default=0)

    def to_numpy(self):
        return np.array([self.energy, self.mac, self.spikes, self.max_spikes, self.params], dtype=np.float)

    def __add__(self, other):
        assert isinstance(other, SPower), "other is not SPower"

        return SPower(
            self.energy + other.energy,
            self.mac + other.mac,
            self.spikes + other.spikes,
            self.max_spikes + other.max_spikes,
            self.params + other.params
        )

    def __mul__(self, other):
        assert isinstance(other, int) or isinstance(other, float), "other is not int"
        return SPower(
            self.energy * other,
            self.mac * other,
            self.spikes * other,
            self.max_spikes * other,
            self.params * other
        )
