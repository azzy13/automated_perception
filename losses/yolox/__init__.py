
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
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_l1 = yolox_l1.Loss(loss_cfg)
        self.loss_cls = yolox_cls.Loss(loss_cfg)
        self.loss_obj = yolox_obj.Loss(loss_cfg)
        self.loss_iou = yolox_iou.Loss(loss_cfg)
        self.use_l1 = False

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
            loss_iou = self.loss_iou(y,(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype),
            )

            loss_obj = self.loss_obj(y,(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype),
            )

            if self.use_l1:
                loss_l1 = self.loss_l1(y,(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype),
            )
            else:
                loss_l1 = 0.0

            loss_cls = self.loss_cls(y,(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype),
            )

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