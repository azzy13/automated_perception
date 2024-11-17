import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import yolox_utils as yolox_utils


class Loss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y_hat, y):
        return self.get_losses(**y_hat, y)
    
    def get_losses(self, imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype, y):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        origin_preds = torch.cat(origin_preds, 1)
        fg_masks = []
        cls_targets = []
        num_fg = 0.0
        num_gts = 0.0
    
        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
        total_num_anchors = outputs.shape[1]

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img) = yolox_utils.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                cls_preds, bbox_preds, obj_preds, labels, imgs)
            cls_target = F.one_hot(
                        gt_matched_classes.to(torch.int64), self.num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
            cls_targets.append(cls_target)
            fg_masks.append(fg_mask)
            

        fg_masks = torch.cat(fg_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        num_fg = max(num_fg, 1)

        loss_cls = (
                self.bcewithlog_loss(
                    cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                )
            ).sum() / num_fg
        return loss_cls

    