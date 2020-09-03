#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 22:23:17 2020

@author: eddie
"""

import torch.nn as nn

class Attention_Module(nn.Module):
    def __init__(self, in_channel):
        super(Attention_Module, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc_block = nn.Sequential(nn.Linear(in_channel, in_channel // 2), 
                                       nn.ReLU(),
                                       nn.Linear(in_channel // 2, in_channel),
                                       nn.Sigmoid())
        
    def forward(self, x):
        batch, cha, w, h = x.shape
        x_v = self.global_avg(x)
        x_v = x_v.view(batch, -1)
        x_cha_att = self.fc_block(x_v)
        x = x.reshape(batch, cha, -1) * x_cha_att.unsqueeze(2)
        x = x.reshape(batch, cha, w, h)
        return x