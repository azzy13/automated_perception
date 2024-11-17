import torch as _torch
import torch.nn.functional as _F

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = _torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = _torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = _torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = _torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = _torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = _torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = _torch.prod(bboxes_a[:, 2:], 1)
        area_b = _torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = _torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

@_torch.no_grad()
def get_assignments(
    self,
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
    mode="gpu",
):

    if mode == "cpu":
        print("------------CPU Mode for This Batch-------------")
        gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
        bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
        gt_classes = gt_classes.cpu().float()
        expanded_strides = expanded_strides.cpu().float()
        x_shifts = x_shifts.cpu()
        y_shifts = y_shifts.cpu()

    img_size = imgs.shape[2:]
    fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        img_size
    )

    bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
    cls_preds_ = cls_preds[batch_idx][fg_mask]
    obj_preds_ = obj_preds[batch_idx][fg_mask]
    num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

    if mode == "cpu":
        gt_bboxes_per_image = gt_bboxes_per_image.cpu()
        bboxes_preds_per_image = bboxes_preds_per_image.cpu()

    pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        
    gt_cls_per_image = (
        _F.one_hot(gt_classes.to(_torch.int64), self.num_classes)
        .float()
        .unsqueeze(1)
        .repeat(1, num_in_boxes_anchor, 1)
    )
    pair_wise_ious_loss = -_torch.log(pair_wise_ious + 1e-8)

    if mode == "cpu":
        cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

    with _torch.cuda.amp.autocast(enabled=False):
        cls_preds_ = (
            cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        )
        pair_wise_cls_loss = _F.binary_cross_entropy(
            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)
    del cls_preds_

    cost = (
        pair_wise_cls_loss
        + 3.0 * pair_wise_ious_loss
        + 100000.0 * (~is_in_boxes_and_center)
    )

    (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
    del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

    if mode == "cpu":
        gt_matched_classes = gt_matched_classes.cuda()
        fg_mask = fg_mask.cuda()
        pred_ious_this_matching = pred_ious_this_matching.cuda()
        matched_gt_inds = matched_gt_inds.cuda()

    return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg,)
    
def get_in_boxes_info(
    self,
    gt_bboxes_per_image,
    expanded_strides,
    x_shifts,
    y_shifts,
    total_num_anchors,
    num_gt,
    img_size
):
    expanded_strides_per_image = expanded_strides[0]
    x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    x_centers_per_image = (
        (x_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
        .repeat(num_gt, 1)
    )  # [n_anchor] -> [n_gt, n_anchor]
    y_centers_per_image = (
         (y_shifts_per_image + 0.5 * expanded_strides_per_image)
        .unsqueeze(0)
        .repeat(num_gt, 1)
    )

    gt_bboxes_per_image_l = (
        (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_r = (
        (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_t = (
        (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_b = (
        (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )

    b_l = x_centers_per_image - gt_bboxes_per_image_l
    b_r = gt_bboxes_per_image_r - x_centers_per_image
    b_t = y_centers_per_image - gt_bboxes_per_image_t
    b_b = gt_bboxes_per_image_b - y_centers_per_image
    bbox_deltas = _torch.stack([b_l, b_t, b_r, b_b], 2)

    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    # in fixed center

    center_radius = 2.5
    # clip center inside image
    gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
    gt_bboxes_per_image_clip[:, 0] = _torch.clamp(gt_bboxes_per_image_clip[:, 0], min=0, max=img_size[1])
    gt_bboxes_per_image_clip[:, 1] = _torch.clamp(gt_bboxes_per_image_clip[:, 1], min=0, max=img_size[0])

    gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)

    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image
    center_deltas = _torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0

    # in boxes and in centers
    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

    is_in_boxes_and_center = (
        is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
    )
    del gt_bboxes_per_image_clip
    return is_in_boxes_anchor, is_in_boxes_and_center

def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
    # Dynamic K
    # ---------------------------------------------------------------
    matching_matrix = _torch.zeros_like(cost)

    ious_in_boxes_matrix = pair_wise_ious
    n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
    topk_ious, _ = _torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
    dynamic_ks = _torch.clamp(topk_ious.sum(1).int(), min=1)
    for gt_idx in range(num_gt):
        _, pos_idx = _torch.topk(
            cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
        )
        matching_matrix[gt_idx][pos_idx] = 1.0

    del topk_ious, dynamic_ks, pos_idx

    anchor_matching_gt = matching_matrix.sum(0)
    if (anchor_matching_gt > 1).sum() > 0:
        cost_min, cost_argmin = _torch.min(cost[:, anchor_matching_gt > 1], dim=0)
        matching_matrix[:, anchor_matching_gt > 1] *= 0.0
        matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
    fg_mask_inboxes = matching_matrix.sum(0) > 0.0
    num_fg = fg_mask_inboxes.sum().item()

    fg_mask[fg_mask.clone()] = fg_mask_inboxes

    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]

    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
        fg_mask_inboxes
    ]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

__all__ = ["bboxes_iou", "get_assignments", "get_in_boxes_info", "dynamic_k_matching"]
