#!/usr/bin/env python
"""PERSIANN-CNN model.

- CNN networks
- Input WV: [32, 32, 1]
- Input IR: [32, 32, 1]

"""
import torch
import torch.nn as nn

class Persiann(nn.Module):
    def __init__(self):
        super().__init__()
        # IR channel convolution
        self.conv_IR = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1),
                                     nn.MaxPool2d(2, 2, 0),
                                     nn.Conv2d(16, 32, 4, 2, 1),
                                     nn.MaxPool2d(2),
                                     nn.ReLU()
                                     )
        # WV channel convolution
        self.conv_WV = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1),
                                     nn.MaxPool2d(2, 2, 0),
                                     nn.Conv2d(16, 32, 4, 2, 1),
                                     nn.MaxPool2d(2),
                                     nn.ReLU()
                                     )
        # Decoder network
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, 5, 2),
                                     nn.ConvTranspose2d(64, 128, 5, 2),
                                     nn.Conv2d(128, 256, 4),
                                     nn.Conv2d(256, 1, 9),
                                     nn.ReLU()
                                     )

    def forward(self, x_IR, x_WV):
        x_IR = self.conv_IR(x_IR)
        x_WV = self.conv_WV(x_WV)
        x = torch.cat((x_IR, x_IR), dim=1)
        x = self.decoder(x)
        return x



# test
x_IR = torch.rand([100, 1, 32, 32])
x_WV = torch.rand([100, 1, 32, 32])

model = Persiann()

model(x_IR, x_WV).shape
# 100, 16, 32, 32