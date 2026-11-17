"""Nes2Net-X backend, copied from Nes2Net/model_scripts/wav2vec2_Nes2Net_X.py.

The module and attribute names are kept identical to the upstream code so that
the backend part of the authors' checkpoint loads with strict=True.
The only change is that this file does not import fairseq.
"""
import math

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return input * self.se(input)


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs, bns, weighted_sum = [], [], []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(kernel_size, 1),
                                   dilation=(dilation, 1), padding=(num_pad, 0)))
            bns.append(nn.BatchNorm2d(width))
            initial_value = torch.ones(1, 1, 1, i + 2) * (1 / (i + 2))
            weighted_sum.append(nn.Parameter(initial_value, requires_grad=True))
        self.weighted_sum = nn.ParameterList(weighted_sum)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, SE_ratio)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out).unsqueeze(-1)  # B C T 1

        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        for i in range(self.nums):
            sp = torch.cat((sp, spx[i]), -1)
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            sp_s = torch.sum(sp * self.weighted_sum[i], dim=-1, keepdim=False)
            out = sp_s if i == 0 else torch.cat((out, sp_s), 1)
        out = torch.cat((out, spx[self.nums].squeeze(-1)), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out


class Nested_Res2Net_TDNN(nn.Module):
    """Nes2Net-X backend with mean pooling (the configuration of the ITW checkpoint)."""

    def __init__(self, Nes_ratio=(8, 8), input_channel=1024, n_output_logits=2,
                 dilation=1, pool_func='mean', SE_ratio=(1,)):
        super().__init__()
        if pool_func != 'mean':
            raise ValueError("only pool_func='mean' is supported (ITW checkpoint)")
        self.Nes_ratio = Nes_ratio[0]
        assert input_channel % Nes_ratio[0] == 0
        C = input_channel // Nes_ratio[0]
        self.C = C
        blocks, bns = [], []
        for _ in range(Nes_ratio[0] - 1):
            blocks.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation,
                                      scale=Nes_ratio[1], SE_ratio=SE_ratio[0]))
            bns.append(nn.BatchNorm1d(C))
        self.Build_in_Res2Nets = nn.ModuleList(blocks)
        self.bns = nn.ModuleList(bns)
        self.bn = nn.BatchNorm1d(input_channel)
        self.relu = nn.ReLU()
        self.pool_func = pool_func
        self.fc = nn.Linear(input_channel, n_output_logits)

    def forward(self, x):  # x: B x 1024 x T
        spx = torch.split(x, self.C, 1)
        for i in range(self.Nes_ratio - 1):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.Build_in_Res2Nets[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[-1]), 1)
        out = self.bn(out)
        out = self.relu(out)
        out = torch.mean(out, dim=-1)
        return self.fc(out)
