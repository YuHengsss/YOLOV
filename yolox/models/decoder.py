# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor
from yolox.models.matcher import HungarianMatcher
from yolox.utils import box_op

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0).transpose(1,2) # [1, batch_size, query_num, hidden_dim]


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt # [query_num, batch_size, hidden_dim]

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)




class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiheadAttention(nn.Module):

    def __init__(self, dim, nhead, attn_drop=0.0, bias=False):
        super().__init__()
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.qk = nn.Linear(dim*3, dim*6, bias=bias)
        self.v_cls = nn.Linear(dim*2, dim*2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
    def forward(self,qk,v_cls,v_loc):
        B, N, C_qk = qk.shape
        B, N, C_cls = v_cls.shape
        B, N, C_loc = v_loc.shape
        qk = self.qk(qk).reshape(B, N, 2, self.num_heads, C_qk // self.num_heads).permute(2, 0, 3, 1, 4)
        v_cls = self.v_cls(v_cls).reshape(B, N, self.num_heads, C_cls // self.num_heads).permute(0, 2, 1, 3)
        v_loc = self.v_loc(v_loc).reshape(B, N, self.num_heads, C_loc // self.num_heads).permute(0, 2, 1, 3)
        q,k = qk[0], qk[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale # [batch_size, num_heads, num_rois, num_rois]
        attn = attn.softmax(dim=-1) #
        attn = self.attn_drop(attn)
        x_cls = (attn @ v_cls).transpose(1, 2).reshape(B, N, C_cls)
        x_loc = (attn @ v_loc).transpose(1, 2).reshape(B, N, C_loc)
        return x_cls, x_loc

class PostAttention(nn.Module):
    def __init__(self, dim, nhead, attn_drop=0.0, bias=False):
        super().__init__()
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.qk = nn.Linear(dim*3, dim*6, bias=bias)
        self.v_cls = nn.Linear(dim*2, dim*2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,qk,v_cls,masks):
        B, N, C_qk = qk.shape
        B, N, C_cls = v_cls.shape
        qk = self.qk(qk).reshape(B, N, 2, self.num_heads, C_qk // self.num_heads).permute(2, 0, 3, 1, 4)
        v_cls = self.v_cls(v_cls).reshape(B, N, self.num_heads, C_cls // self.num_heads).permute(0, 2, 1, 3)
        q,k = qk[0], qk[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale # [batch_size, num_heads, num_rois, num_rois]
        attn = attn.softmax(dim=-1) # [batch_size, num_heads, num_rois, num_rois]

        masks = masks.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn = attn*masks/(self.num_heads*masks.sum(dim=-1,keepdim=True))

        attn = self.attn_drop(attn)
        x_cls = (attn @ v_cls).transpose(1, 2).reshape(B, N, C_cls)
        return x_cls


class PostCrossAttention(nn.Module):
    def __init__(self, dim, nhead, attn_drop=0.0, bias=False):
        super().__init__()
        self.num_heads = nhead
        head_dim = dim // nhead
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim * 3, dim * 3, bias=bias)
        self.k = nn.Linear(dim * 3, dim * 3, bias=bias)
        self.v = nn.Linear(dim * 2, dim * 2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q,k,v,masks):
        B, N, C = q.shape
        B, N, C_v = v.shape
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, N, self.num_heads, C_v // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, num_rois, num_rois]
        attn = attn.softmax(dim=-1)

        masks = masks.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn = attn * masks / (self.num_heads * masks.sum(dim=-1, keepdim=True))

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C_v)
        return x

class TestDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.0, hidden_dim=512):
            super().__init__()
            self.self_attn = PostAttention(d_model, nhead, dropout)
            self.cross_attn = PostCrossAttention(d_model, nhead, dropout)
            self.ffn = FFN(d_model*2, hidden_dim)
            self.norm1 = nn.LayerNorm(d_model*2)
            self.norm2 = nn.LayerNorm(d_model*2)
            self.norm3 = nn.LayerNorm(d_model*2)

    def forward(self,x_cls, x_loc,k_raw,v_raw,masks):
        k_cross = torch.cat([k_raw, x_loc], dim=-1) # [batch_size, num_rois, dim*3]
        x_all = torch.cat([x_cls, x_loc], dim=-1) # [batch_size, num_rois, dim*3]
        attn_cls = self.self_attn(x_all,x_cls,masks) # [batch_size, num_rois, hidden_dim]

        x_cls = attn_cls + x_cls
        x_cls = self.norm1(x_cls)

        q_cross = torch.cat([x_cls, x_loc], dim=-1) # [batch_size, num_rois, dim*3]
        x = self.cross_attn(q_cross,k_cross,v_raw,masks) # [batch_size, num_rois, hidden_dim]
        x = x + x_cls
        x = self.norm2(x)

        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm3(x)
        return  x

class TestFormer(nn.Module):
    def __init__(self, d_model, nhead=4,num_classes=80,hidden_dim=512,layers = 1):
            super().__init__()
            self.decoder_layers = nn.ModuleList([TestDecoderLayer(d_model, nhead,hidden_dim=hidden_dim)
                                                 for i in range(layers)])
            self.layers = layers
            self.class_embed = nn.Linear(d_model*3, num_classes + 1)
            self.bbox_embed = nn.Linear(d_model*3, 4)

    def forward(self,x_cls, x_loc,masks):
        x_cls_raw = x_cls
        for layer in self.decoder_layers:
            x_cls = layer(x_cls, x_loc,x_cls_raw,x_cls_raw,masks)
        x = torch.cat([x_cls, x_loc], dim=-1)
        out_bbox = self.bbox_embed(x).sigmoid()
        out_class = self.class_embed(x)
        out = {'pred_logits': out_class, 'pred_boxes': out_bbox}
        return out

class PostFormer(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers,
                    dim_feedforward=512, dropout=0.1,
                    activation="relu", normalize_before=False,
                    num_queries=100, hidden_dim=512, num_classes=80):
            super().__init__()
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.bbox_embed = nn.Linear(hidden_dim, 4)

            self._reset_parameters()

            self.d_model = d_model
            self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,src,mask,query_embed=None, pos_embed=None):
        '''
        :param src: [batch_size, num_rois, hidden_dim]
        :param mask: [batch_size, num_rois] or None
        :param query_embed: [query_num, hidden_dim]
        :param pos_embed: None or [batch_size, num_rois, hidden_dim]
        :return:
        '''
        if query_embed is None:
            query_embed = self.query_embed.weight
        B,N,C = src.shape
        query_embed = query_embed.unsqueeze(1).repeat(1, B, 1) # [query_num, batch_size, hidden_dim]
        tgt = torch.zeros_like(query_embed) # [query_num, batch_size, hidden_dim]
        src = src.permute(1,0,2) # [num_rois, batch_size, hidden_dim] due to default batch_first=False
        out = self.decoder(tgt, src, pos=pos_embed, query_pos=query_embed, tgt_key_padding_mask=mask)
        #[1, batch_size, query_num,hidden_dim]
        outputs_class = self.class_embed(out) # [1,  batch_size, query_num,num_classes+1]
        outputs_coord = self.bbox_embed(out).sigmoid() # [1, batch_size,query_num, 4]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes=1, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none")
        # prob = F.sigmoid(src_logits) # TODO .detach()
        # p_t = prob * target + (1 - prob) * (1 - target)
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma)
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_op.box_iou(box_op.box_cxcywh_to_xyxy(src_boxes), box_op.box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_op.generalized_box_iou(
            box_op.box_cxcywh_to_xyxy(src_boxes),
            box_op.box_cxcywh_to_xyxy(target_boxes))[0])
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # def loss_masks(self, outputs, targets, indices, num_boxes):
    #     """Compute the losses related to the masks: the focal loss and the dice loss.
    #        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    #     """
    #     assert "pred_masks" in outputs
    #
    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     src_masks = outputs["pred_masks"]
    #     src_masks = src_masks[src_idx]
    #     masks = [t["masks"] for t in targets]
    #     # TODO use valid to mask invalid areas due to padding in loss
    #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #     target_masks = target_masks.to(src_masks)
    #     target_masks = target_masks[tgt_idx]
    #
    #     # upsample predictions to the target size
    #     src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
    #                             mode="bilinear", align_corners=False)
    #     src_masks = src_masks[:, 0].flatten(1)
    #
    #     target_masks = target_masks.flatten(1)
    #     target_masks = target_masks.view(src_masks.shape)
    #     losses = {
    #         "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
    #         "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
    #     }
    #     return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            #'masks': self.loss_masks

            'bce': self.loss_labels_bce,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # TODO: fix me when multi gpu training
        #if is_dist_avail_and_initialized():
            #torch.distributed.all_reduce(num_boxes)
        #num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

