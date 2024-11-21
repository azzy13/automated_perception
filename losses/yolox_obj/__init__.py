#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import yolox_utils as yolox_utils


class Loss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.use_l1 = False
        self.num_classes = 1
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y, y_hat):
        return self.get_losses(y, *y_hat)
    
    def get_losses(
        self,
        y,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        obj_targets = []

        num_fg = 0.0
        num_gts = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                obj_target = outputs.new_zeros((total_num_anchors, 1))
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                
                (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img,) = yolox_utils.get_assignments(
                    self.num_classes,  # noqa
                    batch_idx,
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    bbox_preds,
                    obj_preds,
                    labels,
                    imgs,
                )
                
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                obj_target = fg_mask.unsqueeze(-1)

            obj_targets.append(obj_target.to(dtype))

        obj_targets = torch.cat(obj_targets, 0)
        num_fg = max(num_fg, 1)
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        loss = loss_obj

        return loss
