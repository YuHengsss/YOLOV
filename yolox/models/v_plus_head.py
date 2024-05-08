#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import copy
import time

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger

from yolox.models.post_process import postprocess,get_linking_mat
from yolox.models.post_trans import MSA_yolov, LocalAggregation,visual_attention
from yolox.utils import bboxes_iou
from yolox.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from yolox.utils.debug_vis import visual_predictions
from matplotlib import pyplot as plt
class YOLOVHead(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
            heads=4,
            drop=0.0,
            use_score=True,
            defualt_p=30,
            sim_thresh=0.75,
            pre_nms=0.75,
            ave=True,
            defulat_pre=750,
            test_conf=0.001,
            use_mask=False,
            gmode=True,
            lmode=False,
            both_mode=False,
            localBlocks=1,
            **kwargs
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.Afternum = defualt_p
        self.Prenum = defulat_pre
        self.simN = defualt_p
        self.nms_thresh = pre_nms
        self.n_anchors = 1
        self.use_score = use_score
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.gmode = gmode
        self.lmode = lmode
        self.both_mode = both_mode

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        if kwargs.get('vid_cls',True):
            self.cls_convs2 = nn.ModuleList()
        if kwargs.get('vid_reg',False):
            self.reg_convs2 = nn.ModuleList()

        self.width = int(256 * width)
        self.sim_thresh = sim_thresh
        self.ave = ave
        self.use_mask = use_mask

        if kwargs.get('ota_mode',False):
            if kwargs.get('agg_type','localagg') == 'localagg':
                self.agg =  LocalAggregation(dim=self.width, heads=heads, attn_drop=drop, blocks=localBlocks,
                                                         **kwargs)
                self.cls_pred = nn.Linear(self.width, num_classes)
                self.obj_pred = nn.Linear(self.width, 1)
                self.reg_pred = nn.Linear(self.width, 4)
            elif kwargs.get('agg_type','localagg') == 'msa':
                self.agg = MSA_yolov(dim=self.width, out_dim=4 * self.width,
                                     num_heads=heads, attn_drop=drop, reconf=kwargs.get('reconf',False),)
                if kwargs.get('decouple_reg', False):
                    self.agg_iou = MSA_yolov(dim=self.width, out_dim=4 * self.width,
                                     num_heads=heads, attn_drop=drop, reconf=True)
                self.cls_pred = nn.Linear(4*self.width, num_classes)
                if kwargs.get('reconf', False):
                    self.obj_pred = nn.Linear(4*self.width, 1)
                self.cls_convs2 = nn.ModuleList()
        else:
            if kwargs.get('reconf',False):
                self.agg = LocalAggregation(dim=self.width, heads=heads, attn_drop=drop, blocks=localBlocks,
                                            **kwargs)
                self.cls_pred = nn.Linear(self.width, num_classes)
                self.obj_pred = nn.Linear(self.width, 1)
            else:
                self.agg = MSA_yolov(dim=self.width, out_dim=4 * self.width,
                                     num_heads=heads, attn_drop=drop, reconf=kwargs.get('reconf',False))
                self.cls_pred = nn.Linear(4*self.width, num_classes)
            self.cls_convs2 = nn.ModuleList()

        if both_mode:
            self.g2l = nn.Linear(int(4 * self.width), self.width)
        self.stems = nn.ModuleList()
        self.kwargs = kwargs
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if kwargs.get('vid_cls',True):
                self.cls_convs2.append(
                    nn.Sequential(
                        *[
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                        ]
                    )
                )
            if kwargs.get('vid_reg',False):
                self.reg_convs2.append(
                    nn.Sequential(
                        *[
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                        ]
                    )
                )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.ota_mode = kwargs.get('ota_mode',False)
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5, lframe=0, gframe=32):
        outputs = []
        outputs_decode = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        raw_cls_features = []
        raw_reg_features = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs,  self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            reg_feat = reg_conv(x)
            cls_feat = cls_conv(x)
            if self.kwargs.get('vid_cls',True):
                vid_feat = self.cls_convs2[k](x)
            if self.kwargs.get('vid_reg',False):
                vid_feat_reg = self.reg_convs2[k](x)

            # this part should be the same as the original model
            obj_output = self.obj_preds[k](reg_feat)
            reg_output = self.reg_preds[k](reg_feat)
            cls_output = self.cls_preds[k](cls_feat)
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                outputs.append(output)
            else:
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
               )
            if self.kwargs.get('vid_cls',True):
                raw_cls_features.append(vid_feat)
            else:
                raw_cls_features.append(cls_feat)
            if self.kwargs.get('vid_reg',False):
                raw_reg_features.append(vid_feat_reg)
            else:
                raw_reg_features.append(reg_feat)

            outputs_decode.append(output_decode)
        self.hw = [x.shape[-2:] for x in outputs_decode]
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2
                                   ).permute(0, 2, 1)
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[0].type())
        preds_per_frame = []

        if self.training:
            assigned_packs = self.get_fg_idx(imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                )
            ota_idxs,cls_targets,reg_targets,\
                obj_targets,fg_masks,num_fg,num_gts,l1_targets = assigned_packs

            if not self.ota_mode: ota_idxs = None

        else:
            ota_idxs = None

        cls_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in raw_cls_features], dim=2
        ).permute(0, 2, 1)  # [b,features,channels]
        reg_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in raw_reg_features], dim=2
        ).permute(0, 2, 1)

        pred_result, agg_idx, refine_obj_masks, cls_label_reorder = self.postprocess_widx(decode_res,
                                                    num_classes=self.num_classes,
                                                    nms_thre=self.nms_thresh,
                                                    ota_idxs=ota_idxs,
                                                 )

        for p in agg_idx:
            if p is None: preds_per_frame.append(0)
            else: preds_per_frame.append(p.shape[0])

        if sum(preds_per_frame) == 0 and self.training:
            return torch.tensor(0),0,0,0,0,1,0,0,0

        if not self.training and imgs.shape[0] == 1:
            return pred_result,pred_result



        (features_cls, features_reg, cls_scores,
         fg_scores, locs, all_scores) = self.find_feature_score(cls_feat_flatten,
                                        agg_idx,
                                        reg_feat_flatten,
                                        imgs,
                                        pred_result)
        if features_cls == None and not self.training: return pred_result, pred_result
        if features_cls.shape[0] == 0 and not self.training: return pred_result, pred_result


        features_reg_raw = features_reg.unsqueeze(0)
        features_cls_raw = features_cls.unsqueeze(0)  # [1,features,channels]

        cls_scores = cls_scores.to(cls_feat_flatten.dtype)
        fg_scores = fg_scores.to(cls_feat_flatten.dtype)
        locs = locs.to(cls_feat_flatten.dtype)
        locs = locs.view(1, -1, 4)


        more_args = {'width': imgs.shape[-1], 'height': imgs.shape[-2], 'fg_score': fg_scores,
                     'cls_score': cls_scores, 'all_scores': all_scores, 'lframe': lframe,
                     'afternum': self.Afternum, 'gframe': gframe, 'use_score': self.use_score}

        if self.kwargs.get('agg_type','localagg')=='localagg':
            features_cls, features_reg = self.agg(
                                                features_cls,
                                                features_reg,
                                                locs,
                                                **more_args)


            cls_preds = self.cls_pred(features_cls)
            obj_preds = self.obj_pred(features_reg)
            if self.ota_mode:
                reg_preds = self.reg_pred(features_reg)
                reg_preds = torch.reshape(reg_preds, [-1, 4])
            else:
                reg_preds = None

            cls_preds = torch.reshape(cls_preds, [-1,  self.num_classes])
            obj_preds = torch.reshape(obj_preds, [-1, 1])

        elif self.kwargs.get('agg_type','localagg')=='msa':
            kwargs = self.kwargs
            kwargs.update({'lframe': lframe, 'gframe': gframe, 'afternum': self.Afternum})
            features_cls, features_reg = self.agg(features_cls_raw, features_reg_raw, cls_scores, fg_scores,
                                                 sim_thresh=self.sim_thresh,
                                                 ave=self.ave, use_mask=self.use_mask, **kwargs)
            if self.kwargs.get('decouple_reg',False):
                _,features_reg = self.agg_iou(features_cls_raw, features_reg_raw, cls_scores, fg_scores,
                                                    sim_thresh=self.sim_thresh,
                                                    ave=self.ave, use_mask=self.use_mask, **kwargs)
            cls_preds = self.cls_pred(features_cls)
            if self.kwargs.get('reconf',False):
                obj_preds = self.obj_pred(features_reg)
                reg_preds = None
            else:
                obj_preds, reg_preds = None, None

        if self.training:
            outputs = torch.cat(outputs, 1)
            if not self.ota_mode:
                stime = time.time()
                (refine_cls_targets,
                 refine_cls_masks,
                 refine_obj_targets,
                 refine_obj_masks) = (
                    self.get_iou_based_label(pred_result,agg_idx,labels,outputs,reg_targets,cls_targets)
                )
                refine_cls_targets = torch.cat(refine_cls_targets, 0)
                refine_cls_masks = torch.cat(refine_cls_masks, 0)
                refine_obj_targets = torch.cat(refine_obj_targets, 0)
                refine_obj_masks = torch.cat(refine_obj_masks, 0)
            else:
                refine_cls_targets, refine_cls_masks, refine_obj_targets = None, None, None
                if not self.kwargs.get('vid_ota',False): #use the still object detector supervision
                    #not cat ota idx to the candidates
                    if self.kwargs.get('cls_ota',True):
                        if not self.kwargs.get('cat_ota_fg',True):
                            refine_cls_targets = []
                            for i in range(len(ota_idxs)):
                                tmp_reorder = cls_label_reorder[i]
                                if ota_idxs[i] != None and tmp_reorder!=None and len(tmp_reorder):
                                    tmp_cls_targets = cls_targets[i][torch.stack(tmp_reorder)]
                                    refine_cls_targets.append(tmp_cls_targets)
                            if len(refine_cls_targets):
                                refine_cls_targets = torch.cat(refine_cls_targets, 0)
                            else:
                                refine_cls_targets = torch.cat(cls_targets, 0).new_zeros(0, self.num_classes)
                    else:
                        (refine_cls_targets,
                         refine_cls_masks,
                         _,
                         iou_base_obj_masks) = (
                            self.get_iou_based_label(pred_result, agg_idx, labels, outputs, reg_targets, cls_targets)
                        )
                        refine_cls_targets = torch.cat(refine_cls_targets, 0)
                        refine_cls_masks = torch.cat(refine_cls_masks, 0)
                else:#re-assign lable for vid detections
                    vid_preds = outputs.clone().detach()
                    bidx_accum = 0
                    for b_idx,f_idx in enumerate(agg_idx):
                        if f_idx is None: continue
                        tmp_pred = vid_preds[b_idx,f_idx]
                        #del other preds of the base detector
                        vid_preds[b_idx, :] = -1e3
                        tmp_pred[:,-self.num_classes:] = cls_preds[bidx_accum:bidx_accum+preds_per_frame[b_idx]]
                        tmp_pred[:,4:5] = obj_preds[bidx_accum:bidx_accum+preds_per_frame[b_idx]]
                        vid_preds[b_idx,f_idx] = tmp_pred
                        bidx_accum += preds_per_frame[b_idx]

                    vid_packs = self.get_fg_idx(imgs,
                                                 x_shifts,
                                                 y_shifts,
                                                 expanded_strides,
                                                 labels,
                                                 vid_preds,
                                                 origin_preds,
                                                 dtype=xin[0].dtype,
                                                 )
                    vid_fg_idxs, vid_cls_targets, vid_reg_targets, \
                        vid_obj_targets, vid_fg_masks, vid_num_fg, \
                        vid_num_gts, vid_l1_targets = vid_packs
                    refine_obj_masks,refine_cls_targets = [],[]
                    for b_idx,f_idx in enumerate(agg_idx):
                        if f_idx is None: continue
                        f_idx = f_idx.cuda()
                        refine_obj_masks.append(vid_fg_masks[b_idx][f_idx])
                        tmp_cls_targets = []
                        for feature_idx in f_idx[vid_fg_masks[b_idx][f_idx]]:
                            cls_tar_idx = torch.where(feature_idx==vid_fg_idxs[b_idx])[0]
                            tmp_cls_targets.append(vid_cls_targets[b_idx][cls_tar_idx])
                        if len(tmp_cls_targets):
                            tmp_cls_targets = torch.cat(tmp_cls_targets,0)
                            refine_cls_targets.append(tmp_cls_targets)
                    refine_obj_masks = torch.cat(refine_obj_masks,0)
                    if len(refine_cls_targets):
                        refine_cls_targets = torch.cat(refine_cls_targets, 0)
                    else:
                        refine_cls_targets = torch.cat(cls_targets, 0).new_zeros(0, self.num_classes)


            cls_targets = torch.cat(cls_targets, 0)
            reg_targets = torch.cat(reg_targets, 0)
            obj_targets = torch.cat(obj_targets, 0)
            fg_masks = torch.cat(fg_masks, 0)
            if self.use_l1:
                l1_targets = torch.cat(l1_targets, 0)

            return self.get_losses(
                outputs,
                cls_targets,
                reg_targets,
                obj_targets,
                fg_masks,
                num_fg,
                num_gts,
                l1_targets,
                origin_preds,
                cls_preds,
                obj_preds,
                reg_preds,
                refine_obj_masks,
                refine_cls_targets,
                refine_cls_masks,
                refine_obj_targets,
            )
        else:
            #cls_preds, obj_preds = cls_preds.sigmoid(), obj_preds.sigmoid()
            #refined_preds = torch.cat([reg_preds, obj_preds, cls_preds], 1) #[num_preds, 5+num_classes]

            #split refined_preds into frames according to preds_per_frame which is the number of preds in each frame
            cls_per_frame, obj_per_frame, reg_per_frame = [], [], []
            for i in range(len(preds_per_frame)):
                if self.kwargs.get('reconf',False):
                    obj_per_frame.append(obj_preds[:preds_per_frame[i]].squeeze(-1))
                    obj_preds = obj_preds[preds_per_frame[i]:]
                cls_per_frame.append(cls_preds[:preds_per_frame[i]])
                cls_preds = cls_preds[preds_per_frame[i]:]

            if not self.kwargs.get('reconf',False): obj_per_frame = None
            #obj_per_frame = None
            result, result_ori = postprocess(copy.deepcopy(pred_result),
                                             self.num_classes,
                                             cls_per_frame,
                                             conf_output = obj_per_frame,
                                             nms_thre = nms_thresh,
                                             )
            return result, result_ori  # result

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype, flevel=0):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    def find_feature_score(self, features, idxs, reg_features, imgs=None, predictions=None):
        features_cls = []
        features_reg = []
        cls_scores, all_scores = [], []
        fg_scores = []
        locs = []
        for i, feature in enumerate(features):
            if idxs[i] is None or idxs[i] == []: continue

            features_cls.append(feature[idxs[i]])
            features_reg.append(reg_features[i, idxs[i]])
            cls_scores.append(predictions[i][:, 5])
            fg_scores.append(predictions[i][:, 4])
            locs.append(predictions[i][:, :4])
            all_scores.append(predictions[i][:, -self.num_classes:])
        if len(features_cls) == 0:
            # without any preds
            return None, None, None, None, None, None
        features_cls = torch.cat(features_cls)
        features_reg = torch.cat(features_reg)
        cls_scores = torch.cat(cls_scores)
        fg_scores = torch.cat(fg_scores)
        locs = torch.cat(locs)
        all_scores = torch.cat(all_scores)
        return features_cls, features_reg, cls_scores, fg_scores, locs, all_scores

    def get_losses(
            self,
            outputs,
            cls_targets,
            reg_targets,
            obj_targets,
            fg_masks,
            num_fg,
            num_gts,
            l1_targets,
            origin_preds,
            refined_cls,
            refined_obj,
            refined_reg,
            refined_obj_masks,
            refined_cls_targets,
            refined_cls_masks,
            refined_obj_targets,
    ):

        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        if refined_obj_targets == None:
            refined_obj_targets = refined_obj_masks.type_as(obj_targets)
            refined_obj_targets = refined_obj_targets.view(-1, 1)
            refined_obj_masks = refined_obj_masks.bool().squeeze(-1)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg



        loss_refined_iou = 0

        if self.ota_mode:
            if self.kwargs.get('reconf', True):
                loss_refined_obj = (
                               self.bcewithlog_loss(refined_obj.view(-1, 1), refined_obj_targets)
                           ).sum() / num_fg
            else:loss_refined_obj = 0

            if self.kwargs.get('cls_ota',True):
                loss_refined_cls = (
                                       self.bcewithlog_loss(
                                           refined_cls.view(-1, self.num_classes)[refined_obj_masks], refined_cls_targets
                                       )
                                   ).sum() / num_fg
            else:
                refined_cls_fg = max(float(torch.sum(refined_cls_masks)), 1)
                loss_refined_cls = (
                                       self.bcewithlog_loss(
                                           refined_cls.view(-1, self.num_classes)[refined_cls_masks], refined_cls_targets[refined_cls_masks]
                                       )
                                   ).sum() / refined_cls_fg
        else:
            loss_refined_obj = 0
            refined_cls_fg = max(float(torch.sum(refined_cls_masks)), 1)
            loss_refined_cls = (
                self.bcewithlog_loss(
                    refined_cls.view(-1, self.num_classes)[refined_cls_masks], refined_cls_targets[refined_cls_masks]
                )
            ).sum() / refined_cls_fg
            if self.kwargs.get('reconf',False):
                refined_obj_fg = max(float(torch.sum(refined_obj_masks)), 1)
                loss_refined_obj = (
                    self.bcewithlog_loss(
                        refined_obj.view(-1, 1)[refined_obj_masks], refined_obj_targets[refined_obj_masks]
                    )
                ).sum() / refined_obj_fg
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg

        else:
            loss_l1 = 0.0

        reg_weight = 3.0

        if loss_refined_obj>15:
            #logger.warning('loss_refined_obj is too large: {}'.format(loss_refined_obj))
            #clip the loss
            loss_refined_obj = loss_refined_obj/float(loss_refined_obj)*15
        loss = reg_weight * loss_iou + loss_obj + loss_l1 + loss_cls \
                + loss_refined_cls + reg_weight * loss_refined_iou + loss_refined_obj

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
            loss_refined_cls,
            reg_weight * loss_refined_iou,
            loss_refined_obj,
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
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

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
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
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
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
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 4.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.kwargs.get('vid_dk',10), ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
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

    def postprocess_widx(self, prediction, num_classes, nms_thre=0.5, ota_idxs=None,conf_thresh=0.001):
        # find topK predictions, play the same role as RPN
        '''

        Args:
            prediction: [batch,feature_num,5+clsnum]
            num_classes:
            conf_thre:
            conf_thre_high:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        '''
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        output_index = [None for _ in range(len(prediction))]
        reorder_cls = [None for _ in range(len(prediction))]
        refined_obj_masks = []
        for i, image_pred in enumerate(prediction):
            #take ota idxs as output in training mode
            obj_mask = torch.zeros(0,1)

            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)
            if ota_idxs is not None:
                if len(ota_idxs[i]) > 0 and self.kwargs.get('cat_ota_fg',True):
                    ota_idx = ota_idxs[i]
                    output[i] = detections[ota_idx, :]
                    output_index[i] = ota_idx.cpu()
                    tmp_ota_mask = torch.ones_like(output_index[i]).unsqueeze(1)
                    obj_mask = torch.cat((obj_mask, tmp_ota_mask))

            conf_mask = (detections[:, 4] * detections[:, 5] >= conf_thresh).squeeze()
            minimal_limit = self.kwargs.get('minimal_limit',0)
            maximal_limit = self.kwargs.get('maximal_limit',0)
            if minimal_limit !=0:
                # add a minimum limitation to the number of detections
                if conf_mask.sum() < minimal_limit:
                    #get top minimal_limit detections
                    _, top_idx = torch.topk(detections[:, 4] * detections[:, 5], minimal_limit)
                    conf_mask[top_idx] = True
            if maximal_limit !=0:
                # add a maximum limitation to the number of detections
                if conf_mask.sum() > maximal_limit:
                    logger.warning('current obj above conf thresh: %d' % conf_mask.sum())
                    # #solution 1,
                    _, top_idx = torch.topk(detections[:, 4] * detections[:, 5], maximal_limit)
                    conf_mask = torch.zeros_like(conf_mask)
                    conf_mask[top_idx] = True

                    # #solution 2, only for testing
                    # conf_idx = torch.where(conf_mask)[0]
                    # detections_tmp = detections[conf_mask]
                    # nms_out_index = torchvision.ops.batched_nms(
                    #     detections_tmp[:, :4],
                    #     detections_tmp[:, 4] * detections_tmp[:, 5],
                    #     detections_tmp[:, 6],
                    #     0.9,
                    # )
                    # conf_mask = torch.zeros_like(conf_mask)
                    # conf_mask[conf_idx[nms_out_index]] = True
                    # logger.warning('after nms, current obj above conf thresh: %d' % conf_mask.sum())

            conf_idx = torch.where(conf_mask)[0]
            #logger.info('current obj above conf thresh: %d' % conf_idx.shape[0])
            detections = detections[conf_mask]
            if not detections.size(0):
                refined_obj_masks.append(obj_mask)
                continue

            if self.kwargs.get('use_pre_nms',True):
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )
            else:
                nms_out_index = torch.arange(detections.shape[0])

            #get rid of the idx already in ota_idxs
            if ota_idxs!= None and ota_idxs[i] is not None and len(ota_idxs[i]) > 0:#in ota mode
                if not self.kwargs.get('use_pre_nms',True) :
                    if not self.kwargs.get('cat_ota_fg',True):

                        # not cat ota_idxs, set these ota idx to fg while others to bg
                        obj_mask = torch.zeros_like(nms_out_index).unsqueeze(1)
                        tmp_reorder = []
                        for j in nms_out_index:
                            tmp_idx = torch.where(ota_idxs[i]==conf_idx[j])[0]
                            if len(tmp_idx):
                                obj_mask[j] = 1
                                tmp_reorder.append(tmp_idx[0])
                        #print('total idxs: %d, in ota idxs: %d, total ota idxs %d' % (nms_out_index.shape[0], len(tmp_reorder), ota_idxs[i].shape[0]))
                        reorder_cls[i] = tmp_reorder
                        abs_idx = conf_idx[nms_out_index].cpu()
                    else:
                        #cat ota_idxs and others which is set to bg
                        abs_idx_out_ota = torch.tensor([conf_idx[j] for j in nms_out_index if conf_idx[j] not in ota_idxs[i]])
                        abs_idx = abs_idx_out_ota
                        bg_mask = torch.zeros_like(abs_idx_out_ota).cpu()
                        obj_mask = torch.cat((obj_mask.type_as(bg_mask), bg_mask.unsqueeze(1)))
                else:
                    abs_idx = None
            else:
                abs_idx_out_ota = conf_idx[nms_out_index]
                abs_idx = abs_idx_out_ota.cpu()
                bg_mask = torch.zeros_like(abs_idx_out_ota).cpu()
                obj_mask = torch.cat((obj_mask.type_as(bg_mask), bg_mask.unsqueeze(1)))


            detections = detections[nms_out_index]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

            if output_index[i] is None:
                if self.kwargs.get('use_pre_nms',True):
                    output_index[i] = conf_idx[nms_out_index]
                else:
                    output_index[i] = abs_idx
            else:
                if abs_idx.shape[0] != 0:
                    output_index[i] = torch.cat((output_index[i], abs_idx))

            refined_obj_masks.append(obj_mask)

        if len(refined_obj_masks) > 0:
            refined_obj_masks = torch.cat(refined_obj_masks, 0)
        else:
            refined_obj_masks = torch.zeros(0,1)

        return output, output_index, refined_obj_masks, reorder_cls

    def get_idx_predictions(self,prediction,idxs,num_classes):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)
            output[i] = detections[idxs[i], :]
        return output

    def get_fg_idx(self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        fg_ids = []
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                fg_idx = []#torch.where(fg_mask)[0]
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()

            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [batch,120,class+xywh]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
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
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
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
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                fg_idx = torch.where(fg_mask)[0]

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
            fg_ids.append(fg_idx)

        num_fg = max(num_fg, 1)

        return fg_ids,cls_targets,reg_targets,obj_targets,fg_masks,num_fg,num_gts,l1_targets

    def get_iou_based_label(self,pred_result,idx,labels,outputs,reg_targets,cls_targets):
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
        refine_cls_targets = []
        refine_cls_masks = []
        refine_obj_targets = []
        refine_obj_masks = []
        for batch_idx in range(len(pred_result)):
            num_gt = int(nlabel[batch_idx])
            reg_target = reg_targets[batch_idx]
            if idx[batch_idx] is None: continue
            if num_gt == 0:
                #TODO: handle condition when idx[batch_idx] is None
                refine_cls_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                refine_cls_target[:, -1] = 1 #set no supervision to 1 as flag
                refine_obj_target = outputs.new_zeros((idx[batch_idx].shape[0], 2))
            else:
                refine_cls_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                refine_obj_target = outputs.new_zeros((idx[batch_idx].shape[0], 2))

                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_result[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)
                cls_target = cls_targets[batch_idx]

                refine_obj_target[:, -1] = 1  # set no supervision to 1 as flag
                refine_cls_target[:, -1] = 1  # set no supervision to 1 as flag
                refine_obj_target = refine_obj_target.type_as(reg_target)
                refine_cls_target = refine_cls_target.type_as(cls_target)

                fg_cls_coord = torch.where(max_iou.values >= 0.6)[0]
                bg_coord = torch.where(max_iou.values < 0.3)[0]
                fg_cls_max_idx = max_iou.indices[fg_cls_coord]
                cls_target_onehot = (cls_target > 0).type_as(cls_target)

                fg_ious = max_iou.values[fg_cls_coord].unsqueeze(-1)
                fg_ious = fg_ious.type_as(cls_target)
                refine_cls_target[fg_cls_coord, :self.num_classes] = cls_target_onehot[fg_cls_max_idx, :] * fg_ious
                refine_cls_target[fg_cls_coord,-1] = 0

                refine_obj_target[fg_cls_coord,0] = 1
                refine_obj_target[fg_cls_coord,-1] = 0
                refine_obj_target[bg_coord,0] = 0
                refine_obj_target[bg_coord, -1] = 0

                # for ele_idx, ele in enumerate(idx[batch_idx]):
                #     if max_iou.values[ele_idx] >= 0.6:
                #         max_idx = int(max_iou.indices[ele_idx])
                #         refine_cls_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                #         refine_obj_target[ele_idx,0] = 1
                #     else:
                #         if 0.6>max_iou.values[ele_idx]>0.3:#follow faster rcnn, <0.3 set to bg, >0.6 set to fg, in between set to ignore
                #             refine_obj_target[ele_idx,-1] = 1
                #         refine_cls_target[ele_idx, -1] = 1 # set no supervision to 1 as flag
            refine_cls_targets.append(refine_cls_target[:, :-1])
            refine_obj_targets.append(refine_obj_target[:, :-1])
            refine_cls_masks.append(refine_cls_target[:,-1]==0)
            refine_obj_masks.append(refine_obj_target[:,-1]==0)
        return refine_cls_targets,refine_cls_masks,refine_obj_targets,refine_obj_masks