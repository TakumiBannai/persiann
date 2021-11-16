#!/usr/bin/env python
"""Dataset.

usage;
n_sample = 1000
n_channel = 1
width = 32
height = 32

x1 = torch.rand([n_sample, n_channel, width, height])
x2 = torch.rand([n_sample, n_channel, width, height])
y = torch.rand([n_sample, n_channel, width, height])

dataset = Mydataset(x1, x2, y)
"""

import torch


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x1[index], self.x1[index], self.y[index]
