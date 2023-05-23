#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   snn_lstm.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/30 20:03   lintean      1.0         None
'''
import torch
import torch.nn as nn

from eutils.snn.stbp.functional import ModuleLayer


class SLSTM(nn.LSTM):

    def __init__(self, vth: float = 0.2, snn_process: bool = True, *args, **kwargs):
        super(SLSTM, self).__init__(*args, **kwargs)
        self.vth = vth
        self.snn_process = snn_process
        self.forget_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.input_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.output_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.cell_gate = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.output_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # snn
        self.forget_gate_s = ModuleLayer(self.forget_gate, snn_process=self.snn_process)
        self.input_gate_s = ModuleLayer(self.input_gate, snn_process=self.snn_process)
        self.output_gate_s = ModuleLayer(self.output_gate, snn_process=self.snn_process)
        self.cell_gate_s = ModuleLayer(self.cell_gate, snn_process=self.snn_process)
        self.output_linear_s = ModuleLayer(self.output_linear, snn_process=self.snn_process)
        self.spike = LIFSpike(snn_process=self.snn_process, vth=self.vth)

        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, input_val, hx=None):
        """
        calculate the snn_lstm result
        @param input_val: shape (batch, time, feature, step), feature_size = hidden_size
        @return: output_val: shape (batch, time, feature, step)
        """
        if not self.snn_process:
            return super().forward(input_val, hx)

        batch_size, time_size, feature_size, step_size = input_val.shape
        h = torch.zeros([batch_size, self.hidden_size, step_size], device=self.dummy_param.device)
        c = torch.zeros([batch_size, self.hidden_size, step_size], device=self.dummy_param.device)
        output = []

        for t in range(time_size):
            time_cell = input_val[:, t, :, :]
            c_prev, h_prev = c, h

            x = torch.cat([time_cell, h_prev], dim=1)

            hf = self.forget_gate_s(x)
            hf = self.spike(hf)
            hi = self.input_gate_s(x)
            hi = self.spike(hi)
            ho = self.output_gate_s(x)
            ho = self.spike(ho)
            hc = self.cell_gate_s(x)
            hc = self.spike(hc)

            c = hf * c_prev + hi * hc
            h = ho * c
            y = self.output_linear_s(h)
            y = self.spike(y)

            output.append(y)
        output = torch.stack(output, 1)

        return output, [c, h]
