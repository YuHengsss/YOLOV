#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
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
        )  # cat

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
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


from .swin_transfomer import SwinTransformer
class YOLOPAFPN_Swin(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        width=1,
        depth = 1,
        swin_width=1,
        in_features=(3,4,5),
        in_channels=[512, 1024, 2048],
        out_channels=[256, 512, 1024],
        swin_depth=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        base_dim=96,
        depthwise=False,
        act="relu",
        pretrain_img_size=224,
        ape = False,
        window_size = 7,
    ):
        super().__init__()
        self.backbone = SwinTransformer(out_indices=in_features,depths=swin_depth,num_heads=num_heads,
                                        embed_dim=base_dim,pretrain_img_size=pretrain_img_size,ape=ape,window_size=window_size)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # self.lateral_conv0 = BaseConv(
        #     int(in_channels[2] * width), int(out_channels[1] * width), 1, 1, act=act
        # )
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * swin_width), int(out_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            #int(2 * out_channels[1] * width),
            int(in_channels[1] + out_channels[1] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(out_channels[1] * width), int(out_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            #int(2 * out_channels[0] * width),
            int(in_channels[0]  + out_channels[0] * width),
            int(out_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(out_channels[0] * width), int(out_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * out_channels[0] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(out_channels[1] * width), int(out_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * out_channels[1] * width),
            int(out_channels[2] * width),
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
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 2048->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1536->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 768->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

from .resnet import ResNet
class YOLOPAFPN_ResNet(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        width=1,
        depth = 1,
        resnet_depth=50,
        in_features=("stage3", "stage4", "stage5"),
        in_channels=[512, 1024, 2048],
        out_channels=[256, 512, 1024],
        depthwise=False,
        act="relu",
    ):
        super().__init__()
        self.backbone = ResNet(depth=resnet_depth,out_features=in_features)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(out_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(3 * out_channels[1] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(out_channels[1] * width), int(out_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(3 * out_channels[0] * width),
            int(out_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(out_channels[0] * width), int(out_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * out_channels[0] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(out_channels[1] * width), int(out_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * out_channels[1] * width),
            int(out_channels[2] * width),
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
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 2048->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1536->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 768->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

from .focal import FocalNet
class YOLOPAFPN_focal(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            width=1,
            depth=1,
            focal_width=1,
            focal_depth=1,
            in_features=(3, 4, 5),
            in_channels=[512, 1024, 2048],
            out_channels=[256, 512, 1024],
            depths=[2, 2, 6, 2],
            focal_levels=[4, 4, 4, 4],
            focal_windows=[3, 3, 3, 3],
            use_conv_embed=True,
            use_postln=True,
            use_postln_in_modulation=False,
            use_layerscale=True,
            base_dim=96,
            depthwise=False,
            act="relu",
    ):
        super().__init__()
        self.backbone = FocalNet(embed_dim=base_dim,
                                 depths=depths,
                                 out_indices=in_features,
                                 focal_levels=focal_levels,
                                 focal_windows=focal_windows,
                                 use_conv_embed=use_conv_embed,
                                 use_postln=use_postln,
                                 use_postln_in_modulation=use_postln_in_modulation,
                                 use_layerscale=use_layerscale,
                                 )
        self.in_features = in_features

        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * focal_width), int(out_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(in_channels[1] * focal_width + out_channels[1] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(out_channels[1] * width), int(out_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(in_channels[0]* focal_width + out_channels[0] * width),
            int(out_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(out_channels[0] * width), int(out_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * out_channels[0] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(out_channels[1] * width), int(out_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * out_channels[1] * width),
            int(out_channels[2] * width),
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
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 2048->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1536->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 768->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

        # features = [out_features[f] for f in self.in_features]
        # [x3, x2, x1, x0] = features
        #
        # fpn_out0 = self.lateral_conv0(x0)  # oc4 -> oc3, /16
        # fu_out0 = self.upsample(fpn_out0)  # /8
        # f_out0 = torch.cat([fu_out0, x1], 1)  # oc3 * 2 -> oc3, /8
        # f_out0 = self.C3_p4(f_out0)  # oc3, /8
        #
        # fpn_out1 = self.reduce_conv1(f_out0)  # oc3 -> oc2, /8
        # fu_out1 = self.upsample(fpn_out1)  # oc2, /4
        # f_out1 = torch.cat([fu_out1, x2], 1)  # oc2 * 2 -> oc2, /4
        # pan_out3 = self.C3_p3(f_out1)  # oc2, /4
        #
        # fpn_out2 = self.reduce_conv2(pan_out3)  # oc2 -> oc1, /4
        # f_out2 = self.upsample(fpn_out2)  # oc1, /2
        # f_out2 = torch.cat([f_out2, x3], 1)  # oc1 * 2 -> oc1, /2
        # pan_out2 = self.C3_p2(f_out2)  # oc1, /2
        #
        # p_out1 = self.bu_conv2(pan_out2)  # oc1 -> oc1, /4
        # p_out1 = torch.cat([p_out1, fu_out1], 1)  # oc1 * 2, /4
        # pan_out1 = self.C3_n3(p_out1)  # oc1 -> oc2, /4
        #
        # p_out0 = self.bu_conv1(pan_out1)  # oc2 -> oc2, /4
        # p_out0 = torch.cat([p_out0, fu_out0], 1)  # oc2 * 2 -> oc2, /4
        # pan_out0 = self.C3_n4(p_out0)  # oc2 -> oc3, /8
        #
        # outputs = (pan_out2, pan_out3, pan_out1, pan_out0)
        #
        # return outputs