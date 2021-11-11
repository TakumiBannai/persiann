#!/usr/bin/env python
"""PERSIANN-CNN model.

- CNN networks
- Usage;
x_IR = torch.rand([100, 1, 32, 32])
x_WV = torch.rand([100, 1, 32, 32])
x_cat = torch.rand([100, 64, 8, 8])
model = Persiann()
model(x_IR, x_WV).shape

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

    def forward(self, x_IR, x_WV):
        x_IR = self.conv_IR(x_IR)
        x_WV = self.conv_WV(x_WV)
        x = torch.cat((x_IR, x_IR), dim=1)
        x = self.decoder(x)
        return x



# test



# 100, 64, 16, 16