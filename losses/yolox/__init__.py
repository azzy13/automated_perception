
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import yolox_l1 as yolox_l1
from .. import yolox_utils as yolox_utils

class Loss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init()
        self.loss_cfg = loss_cfg

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
            if self.use_l1:
                origin_preds = torch.cat(origin_preds, 1)

            cls_targets = []
            reg_targets = []
            obj_targets = []
            fg_masks = []

            num_fg = 0.0
            num_gts = 0.0

            for batch_idx in range(outputs.shape[0]):
                num_gt = int(nlabel[batch_idx])
                num_gts += num_gt
                if num_gt == 0:
                    cls_target = outputs.new_zeros((0, self.num_classes))
                    reg_target = outputs.new_zeros((0, 4))
                    obj_target = outputs.new_zeros((total_num_anchors, 1))
                    fg_mask = outputs.new_zeros(total_num_anchors).bool()
                else:
                    gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                    gt_classes = labels[batch_idx, :num_gt, 0]
                    bboxes_preds_per_image = bbox_preds[batch_idx]
                    
                    try:
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                        ) = yolox_utils.get_assignments(  # noqa
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
                    except RuntimeError:
                        print("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                            CPU mode is applied in this batch. If you want to avoid this issue, \
                            try to reduce the batch size or image size.")
                        torch.cuda.empty_cache()
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                        ) = self.get_assignments(  # noqa
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
                            "cpu",
                        )
                    
                    
                    torch.cuda.empty_cache()
                    num_fg += num_fg_img

                    cls_target = F.one_hot(
                        gt_matched_classes.to(torch.int64), self.num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
                    obj_target = fg_mask.unsqueeze(-1)
                    reg_target = gt_bboxes_per_image[matched_gt_inds]


                cls_targets.append(cls_target)
                reg_targets.append(reg_target)
                obj_targets.append(obj_target.to(dtype))
                fg_masks.append(fg_mask)

            cls_targets = torch.cat(cls_targets, 0)
            reg_targets = torch.cat(reg_targets, 0)
            obj_targets = torch.cat(obj_targets, 0)
            fg_masks = torch.cat(fg_masks, 0)

            num_fg = max(num_fg, 1)
            loss_iou = (
                self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
            ).sum() / num_fg
            loss_obj = (
                self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
            ).sum() / num_fg
            loss_cls = (
                self.bcewithlog_loss(
                    cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                )
            ).sum() / num_fg

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