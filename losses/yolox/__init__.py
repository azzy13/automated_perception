
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import yolox_l1 as yolox_l1
from .. import yolox_cls as yolox_cls
from .. import yolox_obj as yolox_obj
from .. import yolox_iou as yolox_iou
from .. import yolox_utils as yolox_utils

class Loss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init()
        self.loss_cfg = loss_cfg
        self.loss_l1 = yolox_l1.Loss(loss_cfg)
        self.loss_cls = yolox_cls.Loss(loss_cfg)
        self.loss_obj = yolox_obj.Loss(loss_cfg)
        self.loss_iou = yolox_iou.Loss(loss_cfg)

    def forward(self, y_hat, y):
        return self.get_losses(**y_hat, y)

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            y
        ):
            loss_iou = self.loss_iou((imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype),
                y,
            )

            loss_obj = self.loss_obj((imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype),
                y,
            )

            loss_cls = self.loss_cls((imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype),
                y,
            )

            if self.use_l1:
                loss_l1 = self.loss_l1((imgs,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    labels,
                    outputs,
                    origin_preds,
                    dtype),
                    y,
                )
            else:
                loss_l1 = 0.0

            reg_weight = 5.0
            loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

            '''return (
                loss,
                reg_weight * loss_iou,
                loss_obj,
                loss_cls,
                loss_l1,
                num_fg / max(num_gts, 1),
            )'''
            return loss