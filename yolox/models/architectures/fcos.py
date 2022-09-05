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


class FCOS(torch.nn.Module):
    def __init__(self, backbone, fpn, head):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def forward(self, x, targets=None):
        res_outs = self.backbone(x)
        fpn_outs, spatial_scale = self.fpn(res_outs)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
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
            outputs = self.head(fpn_outs, targets, x
            )

        return outputs


    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.backbone.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.fpn.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.head.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



