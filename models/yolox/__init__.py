#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from torch import nn
from .models.yolox import YOLOX


class Model(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        self.yolox = YOLOX(training=model_params["training"])

    def forward(self, x, targets=None):
        return self.yolox(x, targets)