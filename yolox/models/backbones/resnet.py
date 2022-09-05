# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from yolox.models.custom_layers import MyDCNv2, ShapeSpec


class NameAdapter(object):
    """Fix the backbones variable names for pretrained weight"""

    def __init__(self, model):
        super(NameAdapter, self).__init__()
        self.model = model

    @property
    def model_type(self):
        return getattr(self.model, '_model_type', '')

    @property
    def variant(self):
        return getattr(self.model, 'variant', '')

    def fix_conv_norm_name(self, name):
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # the naming rule is same as pretrained weight
        if self.model_type == 'SEResNeXt':
            bn_name = name + "_bn"
        return bn_name

    def fix_shortcut_name(self, name):
        if self.model_type == 'SEResNeXt':
            name = 'conv' + name + '_prj'
        return name

    def fix_bottleneck_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            conv_name3 = 'conv' + name + '_x3'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            conv_name3 = name + "_branch2c"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, conv_name3, shortcut_name

    def fix_basicblock_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, shortcut_name

    def fix_layer_warp_name(self, stage_num, count, i):
        name = 'res' + str(stage_num)
        if count > 10 and stage_num == 4:
            if i == 0:
                conv_name = name + "a"
            else:
                conv_name = name + "b" + str(i)
        else:
            conv_name = name + chr(ord("a") + i)
        if self.model_type == 'SEResNeXt':
            conv_name = str(stage_num + 2) + '_' + str(i + 1)
        return conv_name

    def fix_c1_stage_name(self):
        return "res_conv1" if self.model_type == 'ResNeXt' else "conv1"



ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 act=None,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.act = act
        self.dcn_v2 = dcn_v2

        if not self.dcn_v2:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=False)
            self.conv_w_lr = lr
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)
        else:
            self.offset_channel = 2 * filter_size ** 2
            self.mask_channel = filter_size ** 2

            self.conv_offset = nn.Conv2d(
                in_channels=ch_in,
                out_channels=3 * filter_size ** 2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                bias=True)
            # 初始化权重
            torch.nn.init.constant_(self.conv_offset.weight, 0.0)
            torch.nn.init.constant_(self.conv_offset.bias, 0.0)

            # 自实现的DCNv2
            self.conv = MyDCNv2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                dilation=1,
                groups=groups,
                bias=False)
            self.dcn_w_lr = lr
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)

        self.freeze_norm = freeze_norm
        norm_lr = 0. if freeze_norm else lr
        self.norm_lr = norm_lr
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        global_stats = True if freeze_norm else None
        if norm_type in ['sync_bn', 'bn']:
            # ppdet中freeze_norm == True时，use_global_stats = global_stats = True， bn的均值和方差是不会变的！！！，
            # 而且训练时前向传播用的是之前统计均值和方差，而不是当前批次的均值和方差！（即训练时的前向传播就是预测时的前向传播）
            # 所以这里设置momentum = 0.0 让bn的均值和方差不会改变。并且model.train()之后要马上调用model.fix_bn()（让训练bn时的前向传播就是预测时bn的前向传播）
            momentum = 0.0 if freeze_norm else 0.1
            self.norm = nn.BatchNorm2d(ch_out, affine=True, momentum=momentum)
        norm_params = self.norm.parameters()

        if freeze_norm:
            for param in norm_params:
                param.requires_grad_(False)

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)

        if self.norm_type in ['bn', 'sync_bn']:
            out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out

    def fix_bn(self):
        if self.norm is not None:
            if self.freeze_norm:
                self.norm.eval()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if isinstance(self.conv, torch.nn.Conv2d):
            if self.conv.weight.requires_grad:
                param_group_conv = {'params': [self.conv.weight]}
                param_group_conv['lr'] = base_lr * self.conv_w_lr
                param_group_conv['base_lr'] = base_lr * self.conv_w_lr
                param_group_conv['weight_decay'] = base_wd
                param_group_conv['need_clip'] = need_clip
                param_group_conv['clip_norm'] = clip_norm
                param_groups.append(param_group_conv)
        elif isinstance(self.conv, MyDCNv2):   # 自实现的DCNv2
            if self.conv_offset.weight.requires_grad:
                param_group_conv_offset_w = {'params': [self.conv_offset.weight]}
                param_group_conv_offset_w['lr'] = base_lr
                param_group_conv_offset_w['base_lr'] = base_lr
                param_group_conv_offset_w['weight_decay'] = base_wd
                param_group_conv_offset_w['need_clip'] = need_clip
                param_group_conv_offset_w['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_offset_w)
            if self.conv_offset.bias.requires_grad:
                param_group_conv_offset_b = {'params': [self.conv_offset.bias]}
                param_group_conv_offset_b['lr'] = base_lr
                param_group_conv_offset_b['base_lr'] = base_lr
                param_group_conv_offset_b['weight_decay'] = base_wd
                param_group_conv_offset_b['need_clip'] = need_clip
                param_group_conv_offset_b['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_offset_b)
            if self.conv.weight.requires_grad:
                param_group_dcn_weight = {'params': [self.conv.weight]}
                param_group_dcn_weight['lr'] = base_lr * self.dcn_w_lr
                param_group_dcn_weight['base_lr'] = base_lr * self.dcn_w_lr
                param_group_dcn_weight['weight_decay'] = base_wd
                param_group_dcn_weight['need_clip'] = need_clip
                param_group_dcn_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_dcn_weight)
        else:   # 官方DCNv2
            pass
        if self.norm is not None:
            if not self.freeze_norm:
                if self.norm.weight.requires_grad:
                    param_group_norm_weight = {'params': [self.norm.weight]}
                    param_group_norm_weight['lr'] = base_lr * self.norm_lr
                    param_group_norm_weight['base_lr'] = base_lr * self.norm_lr
                    param_group_norm_weight['weight_decay'] = 0.0
                    param_group_norm_weight['need_clip'] = need_clip
                    param_group_norm_weight['clip_norm'] = clip_norm
                    param_groups.append(param_group_norm_weight)
                if self.norm.bias.requires_grad:
                    param_group_norm_bias = {'params': [self.norm.bias]}
                    param_group_norm_bias['lr'] = base_lr * self.norm_lr
                    param_group_norm_bias['base_lr'] = base_lr * self.norm_lr
                    param_group_norm_bias['weight_decay'] = 0.0
                    param_group_norm_bias['need_clip'] = need_clip
                    param_group_norm_bias['clip_norm'] = clip_norm
                    param_groups.append(param_group_norm_bias)


class SELayer(nn.Module):
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        stdv = 1.0 / math.sqrt(ch)
        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(
            ch,
            c_,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

        stdv = 1.0 / math.sqrt(c_)
        self.extract = nn.Linear(
            c_,
            ch,
            weight_attr=paddle.ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=True)

    def forward(self, inputs):
        out = self.pool(inputs)
        out = paddle.squeeze(out, axis=[2, 3])
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = F.sigmoid(out)
        out = paddle.unsqueeze(out, axis=[2, 3])
        scale = out * inputs
        return scale


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(BasicBlock, self).__init__()
        assert groups == 1 and base_width == 64, 'BasicBlock only supports groups=1 and base_width=64'

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_module(
                    'pool',
                    nn.AvgPool2d(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_module(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        # out = paddle.add(x=out, y=short)
        out = out + short
        out = F.relu(out)

        return out

    def fix_bn(self):
        self.branch2a.fix_bn()
        self.branch2b.fix_bn()
        if self.std_senet:
            self.se.fix_bn()
        if not self.shortcut:
            if isinstance(self.short, nn.Sequential):
                for layer in self.short:
                    if isinstance(layer, ConvNormLayer):
                        layer.fix_bn()
            else:
                self.short.fix_bn()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.branch2a.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.branch2b.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.std_senet:
            self.se.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if not self.shortcut:
            if isinstance(self.short, nn.Sequential):
                for layer in self.short:
                    if isinstance(layer, ConvNormLayer):
                        layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            else:
                self.short.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=4,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(BottleNeck, self).__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        # ResNeXt
        width = int(ch_out * (base_width / 64.)) * groups

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_module(
                    'pool',
                    nn.AvgPool2d(
                        kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_module(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out * self.expansion,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out * self.expansion,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=width,
            filter_size=1,
            stride=stride1,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = ConvNormLayer(
            ch_in=width,
            ch_out=width,
            filter_size=3,
            stride=stride2,
            groups=groups,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2)

        self.branch2c = ConvNormLayer(
            ch_in=width,
            ch_out=ch_out * self.expansion,
            filter_size=1,
            stride=1,
            groups=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out * self.expansion)

    def forward(self, inputs):

        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        # out = paddle.add(x=out, y=short)
        out = out + short
        out = F.relu(out)

        return out

    def fix_bn(self):
        self.branch2a.fix_bn()
        self.branch2b.fix_bn()
        self.branch2c.fix_bn()
        if self.std_senet:
            self.se.fix_bn()
        if not self.shortcut:
            if isinstance(self.short, nn.Sequential):
                for layer in self.short:
                    if isinstance(layer, ConvNormLayer):
                        layer.fix_bn()
            else:
                self.short.fix_bn()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.branch2a.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.branch2b.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.branch2c.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.std_senet:
            self.se.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if not self.shortcut:
            if isinstance(self.short, nn.Sequential):
                for layer in self.short:
                    if isinstance(layer, ConvNormLayer):
                        layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            else:
                self.short.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class Blocks(nn.Module):
    def __init__(self,
                 block,
                 ch_in,
                 ch_out,
                 count,
                 name_adapter,
                 stage_num,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(Blocks, self).__init__()

        self.blocks = []
        for i in range(count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            layer = block(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=dcn_v2,
                    std_senet=std_senet)
            self.add_module(conv_name, layer)
            self.blocks.append(layer)
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out

    def fix_bn(self):
        for block in self.blocks:
            block.fix_bn()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for block in self.blocks:
            block.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class ResNet(nn.Module):
    __shared__ = ['norm_type']

    def __init__(self,
                 depth=50,
                 ch_in=64,
                 variant='b',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=True,
                 freeze_at=0,
                 return_idx=[0, 1, 2, 3],
                 dcn_v2_stages=[-1],
                 num_stages=4,
                 std_senet=False):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385

        Args:
            depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
            ch_in (int): output channel of first stage, default 64
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            lr_mult_list (list): learning rate ratio of different resnet stages(2,3,4,5),
                                 lower learning rate ratio is need for pretrained model
                                 got using distillation(default as [1.0, 1.0, 1.0, 1.0]).
            groups (int): group convolution cardinality
            base_width (int): base width of each group convolution
            norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
            norm_decay (float): weight decay for normalization layer weights
            freeze_norm (bool): freeze normalization layers
            freeze_at (int): freeze the backbone at which stage
            return_idx (list): index of the stages whose feature maps are returned
            dcn_v2_stages (list): index of stages who select deformable conv v2
            num_stages (int): total num of stages
            std_senet (bool): whether use senet, default True
        """
        super(ResNet, self).__init__()
        self._model_type = 'ResNet' if groups == 1 else 'ResNeXt'
        assert num_stages >= 1 and num_stages <= 4
        self.depth = depth
        self.variant = variant
        self.groups = groups
        self.base_width = base_width
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(return_idx, Integral):
            return_idx = [return_idx]
        assert max(return_idx) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(return_idx), num_stages)
        self.return_idx = return_idx
        self.num_stages = num_stages
        assert len(lr_mult_list) == 4, \
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages

        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = ResNet_cfg[depth]
        na = NameAdapter(self)

        conv1_name = na.fix_c1_stage_name()
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, conv1_name]]
        self.conv1 = nn.Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.conv1.add_module(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    groups=1,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0))

        self.ch_in = ch_in
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        self._out_channels = [block.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]

        self.res_layers = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            res_layer = Blocks(
                    block,
                    self.ch_in,
                    ch_out_list[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr_mult,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=(i in self.dcn_v2_stages),
                    std_senet=std_senet)
            self.add_module(res_name, res_layer)
            self.res_layers.append(res_layer)
            self.ch_in = self._out_channels[i]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at + 1, num_stages)):
                self._freeze_parameters(self.res_layers[i])

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.requires_grad_(False)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

    def fix_bn(self):
        for layer in self.conv1:
            layer.fix_bn()
        for idx, stage in enumerate(self.res_layers):
            stage.fix_bn()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for layer in self.conv1:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for idx, stage in enumerate(self.res_layers):
            stage.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class Res5Head(nn.Module):
    def __init__(self, depth=50):
        super(Res5Head, self).__init__()
        feat_in, feat_out = [1024, 512]
        if depth < 50:
            feat_in = 256
        na = NameAdapter(self)
        block = BottleNeck if depth >= 50 else BasicBlock
        self.res5 = Blocks(
            block, feat_in, feat_out, count=3, name_adapter=na, stage_num=5)
        self.feat_out = feat_out if depth < 50 else feat_out * 4

    @property
    def out_shape(self):
        return [ShapeSpec(
            channels=self.feat_out,
            stride=16, )]

    def forward(self, roi_feat, stage=0):
        y = self.res5(roi_feat)
        return y
