import torch
import torch.nn as nn
from .. import yolox_utils as yolox_utils


class Loss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.l1_loss = nn.L1Loss(reduction="none")
        self.num_classes = loss_cfg["num_classes"]

    def forward(self, y, y_hat):
        return self.get_losses(y, *y_hat)
    
    def get_losses(self, y,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype):
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

        origin_preds = torch.cat(origin_preds, 1)
        fg_masks = []
        l1_targets = []
        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                l1_target = outputs.new_zeros((0, 4))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = yolox_utils.get_assignments(self.num_classes, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                cls_preds, bbox_preds, obj_preds, labels, imgs)
                l1_target = self.get_l1_target(
                    outputs.new_zeros((num_fg_img, 4)),
                    gt_bboxes_per_image[matched_gt_inds],
                    expanded_strides[0][fg_mask],
                    x_shifts=x_shifts[0][fg_mask],
                    y_shifts=y_shifts[0][fg_mask],
                )
            l1_targets.append(l1_target)
            fg_masks.append(fg_mask)
            

        fg_masks = torch.cat(fg_masks, 0)
        l1_targets = torch.cat(l1_targets, 0)
        num_fg = max(num_fg, 1)

        loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        return loss_l1

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target
    
    