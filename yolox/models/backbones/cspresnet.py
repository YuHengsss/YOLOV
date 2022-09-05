# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.models.backbones.ops import get_act_fn
from yolox.models.backbones.custom_layers import ShapeSpec


class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

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
        if self.bn is not None:
            if self.bn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.bn.weight]}
                param_group_norm_weight['lr'] = base_lr * 1.0
                param_group_norm_weight['base_lr'] = base_lr * 1.0
                param_group_norm_weight['weight_decay'] = 0.0
                param_group_norm_weight['need_clip'] = need_clip
                param_group_norm_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_weight)
            if self.bn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.bn.bias]}
                param_group_norm_bias['lr'] = base_lr * 1.0
                param_group_norm_bias['base_lr'] = base_lr * 1.0
                param_group_norm_bias['weight_decay'] = 0.0
                param_group_norm_bias['need_clip'] = need_clip
                param_group_norm_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_bias)


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if hasattr(self, 'conv'):
            self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        else:
            self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.copy_(kernel)
        self.conv.bias.copy_(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if isinstance(self.fc, torch.nn.Conv2d):
            if self.fc.weight.requires_grad:
                param_group_conv_weight = {'params': [self.fc.weight]}
                param_group_conv_weight['lr'] = base_lr * 1.0
                param_group_conv_weight['base_lr'] = base_lr * 1.0
                param_group_conv_weight['weight_decay'] = base_wd
                param_group_conv_weight['need_clip'] = need_clip
                param_group_conv_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight)
            if self.fc.bias.requires_grad:
                param_group_conv_bias = {'params': [self.fc.bias]}
                param_group_conv_bias['lr'] = base_lr * 1.0
                param_group_conv_bias['base_lr'] = base_lr * 1.0
                param_group_conv_bias['weight_decay'] = base_wd
                param_group_conv_bias['need_clip'] = need_clip
                param_group_conv_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias)


class CSPResStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn='eca'):
        super(CSPResStage, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, shortcut=True)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], 1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if self.conv_down is not None:
            self.conv_down.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for layer in self.blocks:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.attn is not None:
            self.attn.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv3.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class CSPResNet(nn.Module):
    __shared__ = ['width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[0, 1, 2, 3, 4],
                 depth_wise=False,
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 freeze_at=-1,
                 trt=False):
        super(CSPResNet, self).__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act

        if use_large_stem:
            self.stem = nn.Sequential()
            self.stem.add_module('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act))
            self.stem.add_module('conv2', ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act))
            self.stem.add_module('conv3', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))
        else:
            self.stem = nn.Sequential()
            self.stem.add_module('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act))
            self.stem.add_module('conv2', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))

        n = len(channels) - 1
        self.stages = nn.Sequential()
        for i in range(n):
            self.stages.add_module(str(i), CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act))

        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at + 1, n)):
                self._freeze_parameters(self.stages[i])

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.requires_grad_(False)

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for layer in self.stem:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for idx, stage in enumerate(self.stages):
            stage.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]
