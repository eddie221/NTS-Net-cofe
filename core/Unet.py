
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from core.Unet_part import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_up1 = self.up1(x4, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        logits = self.outc(x_up3)
        
        return [x, x1, x2, x3, x4, x_up1, x_up2, x_up3, logits]

if __name__ == "__main__":
    unet = UNet(1,1, True)
    a = torch.randn([1, 1, 224 ,224])
    history = unet(a)
    for item in history:
        print(item.shape)
