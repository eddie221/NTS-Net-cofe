#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 22:22:30 2020

@author: eddie
"""

import torch.nn as nn
import numpy as np
import torch

class cofeature_fast(nn.Module):
    def __init__(self, kernel_size = 3, stride = 1, dilate = 1):
        super(cofeature_fast, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilate = dilate

    def forward(self, x, y = None):
        batch, channel, height, width = x.size()

        center_idxs = [[],[]]
        for y_idx in range(self.kernel_size//2 + self.dilate - 1, height - self.kernel_size//2 - self.dilate + 1, self.stride):
            for x_idx in range(self.kernel_size//2 + self.dilate - 1, width - self.kernel_size//2 - self.dilate + 1, self.stride):
                center_idxs[0].append(y_idx)
                center_idxs[1].append(x_idx)
        center_idxs = np.asarray(center_idxs)
        kernel_count = center_idxs.shape[1]

        center_vector = x[:,:,center_idxs[0],center_idxs[1]]

        center_vector = torch.transpose(center_vector, 1, 2)
        center_vector = center_vector.contiguous().view(batch * kernel_count, channel, 1)
        
        cofe = []
        for y_idx in range(-(self.kernel_size//2 + self.dilate)+1, (self.kernel_size//2 + self.dilate - 1)+1, self.dilate):
            for x_idx in range(-(self.kernel_size//2 + self.dilate)+1, (self.kernel_size//2 + self.dilate - 1)+1, self.dilate):
                #if (y_idx + self.kernel_size//2) * self.kernel_size + x_idx + self.kernel_size//2 <= self.kernel_size * self.kernel_size // 2:
                if (y_idx + self.kernel_size//2) * self.kernel_size + x_idx + self.kernel_size//2 in [0,2,4,6,7,8,10,11,12]:
                    if y is not None:
                        side_vector = y[:,:,center_idxs[0]+y_idx, center_idxs[1]+x_idx]
                    else:
                        side_vector = x[:,:,center_idxs[0]+y_idx, center_idxs[1]+x_idx]
                    side_vector = x[:,:,center_idxs[0]+y_idx, center_idxs[1]+x_idx]
                    side_vector = side_vector.transpose(1,2)
                    side_vector = side_vector.contiguous().view(-1, 1, channel)
    
                    cofeature = torch.bmm(center_vector, side_vector)
                    cofeature = cofeature.view(batch, kernel_count, channel, channel)
    
                    cofeature = torch.sum(cofeature, dim=1, keepdim=False)
                    cofe.append(cofeature)

        cofe = torch.stack(cofe)
        cofe = cofe.transpose(0,1)
        cofe = nn.functional.normalize(cofe, dim=-1)
        return cofe