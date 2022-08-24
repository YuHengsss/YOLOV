#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, other_result=[], targets=None, nms_thresh=0.5, ):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        outputs = self.head(fpn_outs, other_result=other_result, nms_thresh=nms_thresh)

        return outputs
