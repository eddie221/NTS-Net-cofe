#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:50:24 2020

@author: eddie
"""

import core.model as m
import torch

torch.manual_seed(0)
batch_size = 2
PROPOSAL_NUM = 6
model = m.attention_net(PROPOSAL_NUM).cuda()
a = torch.randn([2, 3, 448, 448]).cuda()
label = torch.tensor([0, 7]).cuda()

creterion = torch.nn.CrossEntropyLoss()

raw_logits, concat_logits, part_logits, _, top_n_prob = model(a)
part_loss = m.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                        label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
raw_loss = creterion(raw_logits, label)
concat_loss = creterion(concat_logits, label)
print(part_loss.shape)
rank_loss = m.ranking_loss(top_n_prob, part_loss)
partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
print(total_loss)

