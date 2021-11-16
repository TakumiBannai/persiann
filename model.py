#!/usr/bin/env python
"""PERSIANN-CNN model.

- usage (Input: batch, dim, height, width)
x_ir = torch.rand([100, 1, 32, 32])
x_wv = torch.rand([100, 1, 32, 32])
x_cat = torch.rand([100, 64, 8, 8])
model = Persiann()
out = model(x_ir, x_wv)

"""
import torch
import torch.nn as nn

class Persiann(nn.Module):
    def __init__(self):
        super().__init__()
        # IR channel convolution
        self.conv_IR = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1),
                                     nn.MaxPool2d(2, 2, 0),
                                     nn.Conv2d(16, 32, 3, 1, 1),
                                     nn.MaxPool2d(2, 2, 0),
                                     nn.ReLU()
                                     )
        # WV channel convolution
        self.conv_WV = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1),
                                     nn.MaxPool2d(2, 2, 0),
                                     nn.Conv2d(16, 32, 3, 1, 1),
                                     nn.MaxPool2d(2, 2, 0),
                                     nn.ReLU()
                                     )
        # Decoder network
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                     nn.ConvTranspose2d(64, 128, 4, 2, 1),
                                     nn.Conv2d(128, 256, 3, 1, 1),
                                     nn.Conv2d(256, 1, 9, 1, 4),
                                     nn.ReLU()
                                     )

    def forward(self, x_ir, x_wv):
        x_ir = self.conv_IR(x_ir)
        x_wv = self.conv_WV(x_wv)
        x = torch.cat((x_ir, x_wv), dim=1)
        x = self.decoder(x)
        return x
