#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
from torch import nn
from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX


class Model(nn.Module):
    def __init__(self, training=True):
        super().__init__()
        self.yolox = YOLOX(training=training)

    def forward(self, x, targets=None):
        return self.yolox(x, targets)