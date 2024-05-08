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
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# import paddle.fluid as fluid
# from paddle import ParamAttr
# from paddle.regularizer import L2Decay
# from paddle.nn.initializer import Uniform
# from paddle.nn.initializer import Constant
# from paddle.vision.ops import DeformConv2D


def paddle_yolo_box(conv_output, anchors, stride, num_classes, scale_x_y, im_size, clip_bbox, conf_thresh):
    conv_output = conv_output.permute(0, 2, 3, 1)
    conv_shape       = conv_output.shape
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = conv_output.reshape((batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    rows = T.arange(0, output_size, dtype=T.float32, device=conv_raw_dxdy.device)
    cols = T.arange(0, output_size, dtype=T.float32, device=conv_raw_dxdy.device)
    rows = rows[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis].repeat((1, output_size, 1, 1, 1))
    cols = cols[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis].repeat((1, 1, output_size, 1, 1))
    offset = T.cat([rows, cols], dim=-1)
    offset = offset.repeat((batch_size, 1, 1, anchor_per_scale, 1))
    # Grid Sensitive
    pred_xy = (scale_x_y * T.sigmoid(conv_raw_dxdy) + offset - (scale_x_y - 1.0) * 0.5 ) * stride

    device_name = conv_raw_dwdh.device.type
    device_index = conv_raw_dwdh.device.index
    # _anchors = T.Tensor(anchors, device=exp_wh.device)   # RuntimeError: legacy constructor for device type: cpu was passed device type: cuda, but device type must be: cpu
    _anchors = torch.from_numpy(anchors)
    if device_name == 'cuda':
        _anchors = torch.from_numpy(anchors).cuda(device_index)
    pred_wh = (T.exp(conv_raw_dwdh) * _anchors)

    pred_xyxy = T.cat([pred_xy - pred_wh / 2, pred_xy + pred_wh / 2], dim=-1)   # 左上角xy + 右下角xy
    pred_conf = T.sigmoid(conv_raw_conf)
    # mask = (pred_conf > conf_thresh).float()
    pred_prob = T.sigmoid(conv_raw_prob)
    pred_scores = pred_conf * pred_prob
    # pred_scores = pred_scores * mask
    # pred_xyxy = pred_xyxy * mask

    # paddle中实际的顺序
    pred_xyxy = pred_xyxy.permute(0, 3, 1, 2, 4)
    pred_scores = pred_scores.permute(0, 3, 1, 2, 4)

    pred_xyxy = pred_xyxy.reshape((batch_size, output_size*output_size*anchor_per_scale, 4))
    pred_scores = pred_scores.reshape((batch_size, pred_xyxy.shape[1], num_classes))

    _im_size_h = im_size[:, 0:1]
    _im_size_w = im_size[:, 1:2]
    _im_size = T.cat([_im_size_w, _im_size_h], 1)
    _im_size = _im_size.unsqueeze(1)
    _im_size = _im_size.repeat((1, pred_xyxy.shape[1], 1))
    pred_x0y0 = pred_xyxy[:, :, 0:2] / output_size / stride * _im_size
    pred_x1y1 = pred_xyxy[:, :, 2:4] / output_size / stride * _im_size
    if clip_bbox:
        x0 = pred_x0y0[:, :, 0:1]
        y0 = pred_x0y0[:, :, 1:2]
        x1 = pred_x1y1[:, :, 0:1]
        y1 = pred_x1y1[:, :, 1:2]
        x0 = torch.where(x0 < 0, x0 * 0, x0)
        y0 = torch.where(y0 < 0, y0 * 0, y0)
        x1 = torch.where(x1 > _im_size[:, :, 0:1], _im_size[:, :, 0:1], x1)
        y1 = torch.where(y1 > _im_size[:, :, 1:2], _im_size[:, :, 1:2], y1)
        pred_xyxy = T.cat([x0, y0, x1, y1], -1)
    else:
        pred_xyxy = T.cat([pred_x0y0, pred_x1y1], -1)
    return pred_xyxy, pred_scores

class MyDCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(MyDCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        self.groups = groups

        filter_shape = [out_channels, in_channels // groups, kernel_size, kernel_size]

        self.weight = torch.nn.Parameter(torch.randn(filter_shape))
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels, ))

    def forward(self, x, offset, mask):
        in_C = self.in_channels
        out_C = self.out_channels
        stride = self.stride
        padding = self.padding
        # dilation = self.dilation
        groups = self.groups
        N, _, H, W = x.shape
        _, w_in, kH, kW = self.weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride

        # ================== 1.先对图片x填充得到填充后的图片pad_x ==================
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = torch.zeros((N, in_C, pad_x_H, pad_x_W), dtype=torch.float32, device=x.device)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x

        # ================== 2.求所有采样点的坐标 ==================
        # 卷积核中心点在pad_x中的位置
        y_outer, x_outer = torch.meshgrid([torch.arange(out_H, device=x.device), torch.arange(out_W, device=x.device)])
        y_outer = y_outer * stride + padding
        x_outer = x_outer * stride + padding
        start_pos_yx = torch.stack((y_outer, x_outer), 2).float()       # [out_H, out_W, 2]         仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = start_pos_yx.unsqueeze(0).unsqueeze(3)           # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = torch.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y.requires_grad = False
        start_pos_x.requires_grad = False

        # 卷积核内部的偏移
        half_W = (kW - 1) // 2
        half_H = (kH - 1) // 2
        y_inner2, x_inner2 = torch.meshgrid([torch.arange(kH, device=x.device), torch.arange(kW, device=x.device)])
        y_inner = y_inner2 - half_H
        x_inner = x_inner2 - half_W
        filter_inner_offset_yx = torch.stack((y_inner, x_inner), 2).float()                    # [kH, kW, 2]       卷积核内部的偏移
        filter_inner_offset_yx = torch.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = torch.tile(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_y.requires_grad = False
        filter_inner_offset_x.requires_grad = False

        # 预测的偏移
        offset = offset.permute(0, 2, 3, 1)   # [N, out_H, out_W, kH*kW*2]
        offset_yx = torch.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终采样位置。
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        pos_y = torch.clamp(pos_y, 0.0, H + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内
        pos_x = torch.clamp(pos_x, 0.0, W + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内

        # ================== 3.采样。用F.grid_sample()双线性插值采样。 ==================
        pos_x = pos_x / (pad_x_W - 1) * 2.0 - 1.0
        pos_y = pos_y / (pad_x_H - 1) * 2.0 - 1.0
        xtyt = torch.cat([pos_x, pos_y], -1)  # [N, out_H, out_W, kH*kW, 2]
        xtyt = torch.reshape(xtyt, (N, out_H, out_W * kH * kW, 2))  # [N, out_H, out_W*kH*kW, 2]
        value = F.grid_sample(pad_x, xtyt, mode='bilinear', padding_mode='zeros', align_corners=True)  # [N, in_C, out_H, out_W*kH*kW]
        value = torch.reshape(value, (N, in_C, out_H, out_W, kH * kW))    # [N, in_C, out_H, out_W, kH * kW]
        value = value.permute(0, 1, 4, 2, 3)                              # [N, in_C, kH * kW, out_H, out_W]

        # ================== 4.乘以重要程度 ==================
        # 乘以重要程度
        mask = mask.unsqueeze(1)            # [N,    1, kH * kW, out_H, out_W]
        value = value * mask                # [N, in_C, kH * kW, out_H, out_W]
        new_x = torch.reshape(value, (N, in_C * kH * kW, out_H, out_W))  # [N, in_C * kH * kW, out_H, out_W]

        # ================== 5.乘以本层的权重，加上偏置 ==================
        # 1x1卷积
        rw = torch.reshape(self.weight, (out_C, w_in * kH * kW, 1, 1))  # [out_C, w_in, kH, kW] -> [out_C, w_in*kH*kW, 1, 1]  变成1x1卷积核
        out = F.conv2d(new_x, rw, bias=self.bias, stride=1, groups=groups)  # [N, out_C, out_H, out_W]
        return out


def get_norm(norm_type):
    bn = 0
    sync_bn = 0
    gn = 0
    af = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        sync_bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'in':
        gn = 1
    elif norm_type == 'ln':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    return bn, sync_bn, gn, af


from collections import namedtuple

class ShapeSpec(
        namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super(ShapeSpec, cls).__new__(cls, channels, height, width, stride)


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AffineChannel(torch.nn.Module):
    def __init__(self, num_features):
        super(AffineChannel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_features, ))
        self.bias = torch.nn.Parameter(torch.randn(num_features, ))

    def forward(self, x):
        w = torch.reshape(self.weight, (1, -1, 1, 1))
        b = torch.reshape(self.bias, (1, -1, 1, 1))
        x = x * w + b
        return x


class Conv2dUnit(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 norm_type=None,
                 groups=1,
                 padding=None,
                 norm_groups=32,
                 act=None,
                 freeze_norm=False,
                 norm_decay=0.,
                 lr=1.,
                 bias_lr=None,
                 weight_init=None,
                 bias_init=None,
                 use_dcn=False,
                 name='',
                 data_format='NCHW'):
        super(Conv2dUnit, self).__init__()
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = (filter_size - 1) // 2
        if padding is not None:
            self.padding = padding
        self.act = act
        self.freeze_norm = freeze_norm
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn
        self.name = name
        self.lr = lr

        # conv
        conv_name = name
        self.conv_offset = None
        if use_dcn:
            self.offset_channel = 2 * filter_size**2
            self.mask_channel = filter_size**2
            self.conv_offset = nn.Conv2d(
                in_channels=input_dim,
                out_channels=3 * filter_size**2,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
                bias=True)
            torch.nn.init.constant_(self.conv_offset.weight, 0.0)
            torch.nn.init.constant_(self.conv_offset.bias, 0.0)

            # 自实现的DCNv2
            self.conv = MyDCNv2(
                in_channels=input_dim,
                out_channels=filters,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=bias_attr)
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)
            if bias_attr:
                torch.nn.init.constant_(self.conv.bias, 0.0)
        else:
            self.conv = nn.Conv2d(
                in_channels=input_dim,
                out_channels=filters,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
                groups=groups,
                bias=bias_attr)
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)
            if bias_attr:
                torch.nn.init.constant_(self.conv.bias, 0.0)
                blr = lr
                if bias_lr:
                    blr = bias_lr
                self.blr = blr


        # norm
        assert norm_type in [None, 'bn', 'sync_bn', 'gn', 'affine_channel', 'in', 'ln']
        bn, sync_bn, gn, af = get_norm(norm_type)
        if norm_type == 'in':
            norm_groups = filters
        if norm_type == 'ln':
            norm_groups = 1
        if conv_name == "conv1":
            norm_name = "bn_" + conv_name
            if gn:
                norm_name = "gn_" + conv_name
            if af:
                norm_name = "af_" + conv_name
        else:
            norm_name = "bn" + conv_name[3:]
            if gn:
                norm_name = "gn" + conv_name[3:]
            if af:
                norm_name = "af" + conv_name[3:]
        self.bn = None
        self.gn = None
        self.af = None
        if bn:
            self.bn = torch.nn.BatchNorm2d(filters)
            torch.nn.init.constant_(self.bn.weight, 1.0)
            torch.nn.init.constant_(self.bn.bias, 0.0)
        if sync_bn:
            self.bn = torch.nn.BatchNorm2d(filters)
            torch.nn.init.constant_(self.bn.weight, 1.0)
            torch.nn.init.constant_(self.bn.bias, 0.0)
            # self.bn = torch.nn.SyncBatchNorm(filters, weight_attr=pattr, bias_attr=battr)
        if gn:
            self.gn = torch.nn.GroupNorm(num_groups=norm_groups, num_channels=filters)
            torch.nn.init.constant_(self.gn.weight, 1.0)
            torch.nn.init.constant_(self.gn.bias, 0.0)
        if af:
            self.af = AffineChannel(filters)
            torch.nn.init.constant_(self.af.weight, 1.0)
            torch.nn.init.constant_(self.af.bias, 0.0)

        # act
        self.act = None
        if act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky':
            self.act = torch.nn.LeakyReLU(0.1)
        elif act == 'mish':
            self.act = Mish()
        elif act is None:
            pass
        else:
            raise NotImplementedError("Activation \'{}\' is not implemented.".format(act))


    def freeze(self):
        if self.conv is not None:
            if self.conv.weight is not None:
                self.conv.weight.requires_grad = False
            if self.conv.bias is not None:
                self.conv.bias.requires_grad = False
        if self.conv_offset is not None:
            if self.conv_offset.weight is not None:
                self.conv_offset.weight.requires_grad = False
            if self.conv_offset.bias is not None:
                self.conv_offset.bias.requires_grad = False
        if self.bn is not None:
            self.bn.weight.requires_grad = False
            self.bn.bias.requires_grad = False
        if self.gn is not None:
            self.gn.weight.requires_grad = False
            self.gn.bias.requires_grad = False
        if self.af is not None:
            self.af.weight.requires_grad = False
            self.af.bias.requires_grad = False

    def fix_bn(self):
        if self.bn is not None:
            self.bn.eval()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if isinstance(self.conv, torch.nn.Conv2d):
            if self.conv.weight.requires_grad:
                param_group_conv = {'params': [self.conv.weight]}
                param_group_conv['lr'] = base_lr * self.lr
                param_group_conv['base_lr'] = base_lr * self.lr
                param_group_conv['weight_decay'] = base_wd
                param_group_conv['need_clip'] = need_clip
                param_group_conv['clip_norm'] = clip_norm
                param_groups.append(param_group_conv)
                if self.conv.bias is not None:
                    if self.conv.bias.requires_grad:
                        param_group_conv_bias = {'params': [self.conv.bias]}
                        param_group_conv_bias['lr'] = base_lr * self.blr
                        param_group_conv_bias['base_lr'] = base_lr * self.blr
                        param_group_conv_bias['weight_decay'] = 0.0
                        param_group_conv_bias['need_clip'] = need_clip
                        param_group_conv_bias['clip_norm'] = clip_norm
                        param_groups.append(param_group_conv_bias)
        elif isinstance(self.conv, MyDCNv2):   # 自实现的DCNv2
            if self.conv_offset.weight.requires_grad:
                param_group_conv_offset_w = {'params': [self.conv_offset.weight]}
                param_group_conv_offset_w['lr'] = base_lr * self.lr
                param_group_conv_offset_w['base_lr'] = base_lr * self.lr
                param_group_conv_offset_w['weight_decay'] = base_wd
                param_group_conv_offset_w['need_clip'] = need_clip
                param_group_conv_offset_w['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_offset_w)
            if self.conv_offset.bias.requires_grad:
                param_group_conv_offset_b = {'params': [self.conv_offset.bias]}
                param_group_conv_offset_b['lr'] = base_lr * self.lr
                param_group_conv_offset_b['base_lr'] = base_lr * self.lr
                param_group_conv_offset_b['weight_decay'] = base_wd
                param_group_conv_offset_b['need_clip'] = need_clip
                param_group_conv_offset_b['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_offset_b)
            if self.conv.weight.requires_grad:
                param_group_dcn_weight = {'params': [self.conv.weight]}
                param_group_dcn_weight['lr'] = base_lr * self.lr
                param_group_dcn_weight['base_lr'] = base_lr * self.lr
                param_group_dcn_weight['weight_decay'] = base_wd
                param_group_dcn_weight['need_clip'] = need_clip
                param_group_dcn_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_dcn_weight)
        else:   # 官方DCNv2
            pass
        if self.bn is not None:
            if self.bn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.bn.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_group_norm_weight['need_clip'] = need_clip
                param_group_norm_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_weight)
            if self.bn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.bn.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_group_norm_bias['need_clip'] = need_clip
                param_group_norm_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_bias)
        if self.gn is not None:
            if self.gn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.gn.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_group_norm_weight['need_clip'] = need_clip
                param_group_norm_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_weight)
            if self.gn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.gn.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_group_norm_bias['need_clip'] = need_clip
                param_group_norm_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_bias)
        if self.af is not None:
            if self.af.weight.requires_grad:
                param_group_norm_weight = {'params': [self.af.weight]}
                param_group_norm_weight['lr'] = base_lr * self.lr
                param_group_norm_weight['base_lr'] = base_lr * self.lr
                param_group_norm_weight['weight_decay'] = 0.0
                param_group_norm_weight['need_clip'] = need_clip
                param_group_norm_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_weight)
            if self.af.bias.requires_grad:
                param_group_norm_bias = {'params': [self.af.bias]}
                param_group_norm_bias['lr'] = base_lr * self.lr
                param_group_norm_bias['base_lr'] = base_lr * self.lr
                param_group_norm_bias['weight_decay'] = 0.0
                param_group_norm_bias['need_clip'] = need_clip
                param_group_norm_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_bias)

    def forward(self, x):
        if self.use_dcn:
            offset_mask = self.conv_offset(x)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = T.sigmoid(mask)
            x = self.conv(x, offset, mask=mask)
        else:
            x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.af:
            x = self.af(x)
        if self.act:
            x = self.act(x)
        return x


def batch_norm(ch,
               norm_type='bn',
               norm_decay=0.,
               freeze_norm=False,
               initializer=None,
               data_format='NCHW'):
    norm_lr = 0. if freeze_norm else 1.

    if norm_type in ['sync_bn', 'bn']:
        norm_layer = nn.BatchNorm2d(ch, affine=True)

    norm_params = norm_layer.parameters()
    if freeze_norm:
        for param in norm_params:
            param.requires_grad_(False)

    return norm_layer


class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 norm_type='bn',
                 norm_decay=0.,
                 act="leaky",
                 freeze_norm=False,
                 data_format='NCHW',
                 name=''):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.batch_norm = batch_norm(
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(out, 0.1)
        else:
            out = getattr(F, self.act)(out)
        return out

    def fix_bn(self):
        if self.batch_norm is not None:
            if self.freeze_norm:
                self.batch_norm.eval()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if isinstance(self.conv, torch.nn.Conv2d):
            if self.conv.weight.requires_grad:
                param_group_conv = {'params': [self.conv.weight]}
                param_group_conv['lr'] = base_lr * 1.0
                param_group_conv['base_lr'] = base_lr * 1.0
                param_group_conv['weight_decay'] = base_wd
                param_group_conv['need_clip'] = need_clip
                param_group_conv['clip_norm'] = clip_norm
                param_groups.append(param_group_conv)
        if self.batch_norm is not None:
            if not self.freeze_norm:
                if self.batch_norm.weight.requires_grad:
                    param_group_norm_weight = {'params': [self.batch_norm.weight]}
                    param_group_norm_weight['lr'] = base_lr * 1.0
                    param_group_norm_weight['base_lr'] = base_lr * 1.0
                    param_group_norm_weight['weight_decay'] = self.norm_decay
                    param_group_norm_weight['need_clip'] = need_clip
                    param_group_norm_weight['clip_norm'] = clip_norm
                    param_groups.append(param_group_norm_weight)
                if self.batch_norm.bias.requires_grad:
                    param_group_norm_bias = {'params': [self.batch_norm.bias]}
                    param_group_norm_bias['lr'] = base_lr * 1.0
                    param_group_norm_bias['base_lr'] = base_lr * 1.0
                    param_group_norm_bias['weight_decay'] = self.norm_decay
                    param_group_norm_bias['need_clip'] = need_clip
                    param_group_norm_bias['clip_norm'] = clip_norm
                    param_groups.append(param_group_norm_bias)



class CoordConv2(torch.nn.Module):
    def __init__(self, coord_conv=True):
        super(CoordConv2, self).__init__()
        self.coord_conv = coord_conv

    def forward(self, input):
        if not self.coord_conv:
            return input
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        x_range = T.arange(0, w, dtype=T.float32, device=input.device) / (w - 1) * 2.0 - 1
        y_range = T.arange(0, h, dtype=T.float32, device=input.device) / (h - 1) * 2.0 - 1
        x_range = x_range[np.newaxis, np.newaxis, np.newaxis, :].repeat((b, 1, h, 1))
        y_range = y_range[np.newaxis, np.newaxis, :, np.newaxis].repeat((b, 1, 1, w))
        offset = T.cat([input, x_range, y_range], dim=1)
        return offset


def add_coord(x, data_format):
    b = x.shape[0]
    if data_format == 'NCHW':
        h, w = x.shape[2], x.shape[3]
    else:
        h, w = x.shape[1], x.shape[2]

    gx = T.arange(0, w, dtype=x.dtype, device=x.device) / (w - 1.) * 2.0 - 1.
    gy = T.arange(0, h, dtype=x.dtype, device=x.device) / (h - 1.) * 2.0 - 1.

    if data_format == 'NCHW':
        gx = gx.reshape([1, 1, 1, w]).expand([b, 1, h, w])
        gy = gy.reshape([1, 1, h, 1]).expand([b, 1, h, w])
    else:
        gx = gx.reshape([1, 1, w, 1]).expand([b, h, w, 1])
        gy = gy.reshape([1, h, 1, 1]).expand([b, h, w, 1])

    gx.requires_grad = False
    gy.requires_grad = False
    return gx, gy


class CoordConv(torch.nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 padding,
                 norm_type,
                 freeze_norm=False,
                 name='',
                 data_format='NCHW'):
        """
        PPYOLO专用的CoordConv，强制绑定一个Conv + BN + LeakyRELU.
        CoordConv layer, see https://arxiv.org/abs/1807.03247

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            name (str): layer name
            data_format (str): data format, NCHW or NHWC

        """
        super(CoordConv, self).__init__()
        self.conv = ConvBNLayer(
            ch_in + 2,
            ch_out,
            filter_size=filter_size,
            padding=padding,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            data_format=data_format,
            name=name)
        self.data_format = data_format

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)

    def forward(self, x):
        gx, gy = add_coord(x, self.data_format)
        if self.data_format == 'NCHW':
            y = torch.cat([x, gx, gy], 1)
        else:
            y = torch.cat([x, gx, gy], -1)
        y = self.conv(y)
        return y


class SPP2(torch.nn.Module):
    def __init__(self, seq='asc'):
        super(SPP2, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, 1, 2)
        x_3 = F.max_pool2d(x, 9, 1, 4)
        x_4 = F.max_pool2d(x, 13, 1, 6)
        if self.seq == 'desc':
            out = torch.cat([x_4, x_3, x_2, x_1], dim=1)
        else:
            out = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        return out


class SPP(torch.nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 norm_type='bn',
                 freeze_norm=False,
                 name='',
                 act='leaky',
                 data_format='NCHW'):
        """
        PPYOLO专用的SPP，强制绑定一个Conv + BN + LeakyRELU.
        SPP layer, which consist of four pooling layer follwed by conv layer

        Args:
            ch_in (int): input channel of conv layer
            ch_out (int): output channel of conv layer
            k (int): kernel size of conv layer
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            act (str): activation function
            data_format (str): data format, NCHW or NHWC
        """
        super(SPP, self).__init__()
        self.pool = torch.nn.ModuleList()
        self.data_format = data_format
        for size in pool_size:
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.pool.append(pool)
        self.conv = ConvBNLayer(
            ch_in,
            ch_out,
            k,
            padding=k // 2,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            name=name,
            act=act,
            data_format=data_format)

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == "NCHW":
            y = torch.cat(outs, 1)
        else:
            y = torch.cat(outs, -1)

        y = self.conv(y)
        return y


class DropBlock2(torch.nn.Module):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9):
        super(DropBlock2, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, input):
        if not self.training:
            return input

        def CalculateGamma(input, block_size, keep_prob):
            h = input.shape[2]  # int
            h = np.array([h])
            h = torch.tensor(h, dtype=torch.float32, device=input.device)
            feat_shape_t = h.reshape((1, 1, 1, 1))  # shape: [1, 1, 1, 1]
            feat_area = torch.pow(feat_shape_t, 2)  # shape: [1, 1, 1, 1]

            block_shape_t = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=input.device) + block_size
            block_area = torch.pow(block_shape_t, 2)

            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = torch.pow(useful_shape_t, 2)

            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output

        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = input.shape
        p = gamma.repeat(input_shape)

        input_shape_tmp = input.shape
        random_matrix = torch.rand(input_shape_tmp, device=input.device)
        one_zero_m = (random_matrix < p).float()

        mask_flag = torch.nn.functional.max_pool2d(one_zero_m, (self.block_size, self.block_size), stride=1, padding=1)
        mask = 1.0 - mask_flag

        elem_numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        elem_numel_m = float(elem_numel)

        elem_sum = mask.sum()

        output = input * mask * elem_numel_m / elem_sum
        return output


class DropBlock(torch.nn.Module):
    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = torch.rand(x.shape, device=x.device)
            matrix = (matrix < gamma).float()
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y
        # return x


# class PointGenerator(object):
#
#     def _meshgrid(self, x, y, w, h, row_major=True):
#         xx = paddle.tile(paddle.reshape(x, (1, -1)), [h, 1])
#         yy = paddle.tile(paddle.reshape(y, (-1, 1)), [1, w])
#
#         xx = paddle.reshape(xx, (-1, ))
#         yy = paddle.reshape(yy, (-1, ))
#         if row_major:
#             return xx, yy
#         else:
#             return yy, xx
#
#     def grid_points(self, featmap_size, stride=16):
#         feat_h, feat_w = featmap_size
#         eps = 1e-3
#         shift_x = paddle.arange(0., feat_w - eps, 1., dtype='float32') * stride
#         shift_y = paddle.arange(0., feat_h - eps, 1., dtype='float32') * stride
#
#         shift_xx, shift_yy = self._meshgrid(shift_x, shift_y, feat_w, feat_h)
#         stride = paddle.full(shape=shift_xx.shape, fill_value=stride, dtype='float32')
#         all_points = paddle.stack([shift_xx, shift_yy, stride], axis=-1)
#         return all_points
#
#     def valid_flags(self, featmap_size, valid_size, device='cuda'):
#         # feat_h, feat_w = featmap_size
#         # valid_h, valid_w = valid_size
#         # assert valid_h <= feat_h and valid_w <= feat_w
#         # valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
#         # valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
#         # valid_x[:valid_w] = 1
#         # valid_y[:valid_h] = 1
#         # valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
#         # valid = valid_xx & valid_yy
#         # return valid
#         pass
#
#


