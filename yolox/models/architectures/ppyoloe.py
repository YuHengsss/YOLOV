#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import torch


class PPYOLOE(torch.nn.Module):
    def __init__(self, backbone, neck, yolo_head):
        super(PPYOLOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = yolo_head

    def forward(self, x, targets=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        if self.training:
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_feats, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs =  self.head(
                fpn_feats, targets, x
            )
        return outputs

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.backbone.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.neck.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.head.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



