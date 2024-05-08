# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import math
import torch
from torch import nn
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
    device = position_mat.device
    # position_mat, [num_rois, num_nongt_rois, 4]
    feat_range = torch.arange(0, feat_dim / 8, device=device)
    dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

    position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
    position_mat = position_mat * 100.0

    div_mat = position_mat / dim_mat
    sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

    # [num_rois, num_nongt_rois, 4, feat_dim / 4]
    embedding = torch.cat([sin_mat, cos_mat], dim=3)
    # [num_rois, num_nongt_rois, feat_dim]
    embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

    return embedding

def extract_position_matrix(bbox, ref_bbox):
    xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1) # ref_bbox: [num_rois, 4]
    bbox_width_ref = xmax - xmin + 1
    bbox_height_ref = ymax - ymin + 1
    center_x_ref = 0.5 * (xmin + xmax) # [num_rois, 1]
    center_y_ref = 0.5 * (ymin + ymax) # [num_rois, 1]

    xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    delta_x = center_x - center_x_ref.transpose(0, 1) # [num_rois, num_nongt_rois]
    delta_x = delta_x / bbox_width
    delta_x = (delta_x.abs() + 1e-3).log()

    delta_y = center_y - center_y_ref.transpose(0, 1)
    delta_y = delta_y / bbox_height
    delta_y = (delta_y.abs() + 1e-3).log()

    delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
    delta_width = delta_width.log()

    delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
    delta_height = delta_height.log()

    position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

    return position_matrix

def pure_position_embedding(rois,width,height):
    xmin, ymin, xmax, ymax = torch.chunk(rois, 4, dim=1)
    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    delta_x = center_x / width # [num_rois, 1]
    delta_x = (delta_x.abs() + 1e-3).log()

    delta_y = center_y / height
    delta_y = (delta_y.abs() + 1e-3).log()

    delta_width = bbox_width / width
    delta_width = delta_width.log()

    delta_height = bbox_height / height
    delta_height = delta_height.log()

    position_matrix = torch.cat([delta_x, delta_y, delta_width, delta_height], dim=1) # [num_rois, 4]
    return position_matrix

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area,iou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones([x.shape[0],x.shape[-2],x.shape[-1]],dtype=torch.bool,device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # [batch_size, h, w]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # [64]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # [64]

        pos_x = x_embed[:, :, :, None] / dim_t # [batch_size, h, w, 64]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # [batch_size, 64*2, h, w]
        return pos





