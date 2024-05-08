#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknetP6 as CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5", "dark6"),
        in_channels=[256, 512, 1024, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv_1 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act
        )
        self.C3_p5 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv0 = Conv(
            int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act
        )
        self.C3_n5 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features
        #dark 3,4,5,6
        fpn_out0 = self.lateral_conv_1(x0)  # 1024->1024/64 dark6
        f_out0 = self.upsample(fpn_out0)  # 1024/32
        f_out0 = torch.cat([f_out0, x1], 1)  # 1024->2048/32
        f_out0 = self.C3_p5(f_out0)  # 2048->1024/32

        fpn_out1 = self.lateral_conv0(f_out0)  # 1024->512/32
        f_out1 = self.upsample(fpn_out1)  # 512/16
        f_out1 = torch.cat([f_out1, x2], 1)  # 512->1024/16
        f_out1 = self.C3_p4(f_out1)  # 1024->512/16

        fpn_out2 = self.reduce_conv1(f_out1) #512->256/16
        f_out2 = self.upsample(fpn_out2) #256/8
        f_out2 = torch.cat([f_out2, x3], 1)  # 256->512/8
        pan_out3 = self.C3_p3(f_out2) #512/8-->256/8

        p_out2 = self.bu_conv2(pan_out3)  # 256->256/16
        p_out2 = torch.cat([p_out2, fpn_out2], 1)  # 256->512/16
        pan_out2 = self.C3_n3(p_out2)  # 512->512/16

        p_out1 = self.bu_conv1(pan_out2)  # 512->512/32
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 512->1024/32
        pan_out1 = self.C3_n4(p_out1)  # 1024->1024/32


        p_out0 = self.bu_conv0(pan_out1)  # 1024->1024/64
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 1024->1024/64
        pan_out0 = self.C3_n5(p_out0)  # 1024->1024/32



        outputs = (pan_out3, pan_out2, pan_out1, pan_out0)
        return outputs
