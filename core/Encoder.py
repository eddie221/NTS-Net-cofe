#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:33:03 2020

@author: eddie
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Encoder(nn.Module):
    
    def __init__(self, in_cha):
        super(Encoder, self).__init__()
        self.inc = DoubleConv(in_cha, 64)
        self.conv1 = Down(64, 128)
        self.conv2 = Down(128, 256)
        self.conv3 = Down(256, 512)
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.inc(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_mean = torch.mean(x, dim = 1, keepdim = True)
        x_threshold = torch.mean(x_mean.view(B, x_mean.shape[1], -1), dim = 2, keepdim = True).unsqueeze(3)
        x_mean = F.interpolate(x_mean, size = (H, W), mode = 'bilinear', align_corners = True)
        x_mask = (x_mean > x_threshold).float()
        
        return x_mask
        
        
if __name__ == "__main__":
    torch.manual_seed(0)
    model = Encoder(3,)
    a = torch.randn([1,3,224,224])
    model(a)

