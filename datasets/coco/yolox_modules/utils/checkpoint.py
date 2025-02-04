#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch

import os
import shutil


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)
