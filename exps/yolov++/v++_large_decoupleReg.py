import os
import torch.nn as nn
import sys
import torch
sys.path.append("..")
from exps.yolov.yolov_base import Exp as MyExp
from loguru import logger

#exp after OTA_VID_woRegScore, exp 8 in the doc, decouple the reg and cls refinement
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0  # 1#0.67
        self.width = 1.0  # 1#0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path

        self.warmup_epochs = 0
        self.no_aug_epochs = 2
        self.pre_no_aug = 2
        self.eval_interval = 1
        self.gmode = True
        self.lmode = False
        self.lframe = 0
        self.lframe_val = 0
        self.gframe = 16
        self.gframe_val = 32
        self.use_loc_emd = False
        self.iou_base = False
        self.reconf = True
        self.loc_fuse_type = 'identity'
        self.output_dir = "./V++_outputs"
        self.stem_lr_ratio = 0.1
        self.ota_mode = True
        #check pre_nms for testing when use_pre_nms is False in training: Result: AP50 drop 3.0
        self.use_pre_nms = False
        self.cat_ota_fg = False
        self.agg_type='msa'
        self.minimal_limit = 50
        self.decouple_reg = True

    def get_model(self):
        # rewrite get model func from yolox
        if self.backbone_name == 'MCSP':
            in_channels = [256, 512, 1024]
            from yolox.models import YOLOPAFPN
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        elif 'Swin' in self.backbone_name:
            from yolox.models import YOLOPAFPN_Swin

            if self.backbone_name == 'Swin_Tiny':
                in_channels = [192, 384, 768]
                out_channels = [192, 384, 768]
                backbone = YOLOPAFPN_Swin(in_channels=in_channels,
                                          out_channels=out_channels,
                                          act=self.act,
                                          in_features=(1, 2, 3))
            elif self.backbone_name == 'Swin_Base':
                in_channels = [256, 512, 1024]
                out_channels = [256, 512, 1024]
                backbone = YOLOPAFPN_Swin(in_channels=in_channels,
                                          out_channels=out_channels,
                                          act=self.act,
                                          in_features=(1, 2, 3),
                                          swin_depth=[2, 2, 18, 2],
                                          num_heads=[4, 8, 16, 32],
                                          base_dim=int(in_channels[0] / 2),
                                          pretrain_img_size=self.pretrain_img_size,
                                          window_size=self.window_size,
                                          width=self.width,
                                          depth=self.depth
                                          )
        elif 'Focal' in self.backbone_name:
            from yolox.models import YOLOPAFPN_focal
            fpn_in_channles = [96 * 4, 96 * 8, 96 * 16]
            in_channels = self.focal_fpn_channels
            backbone = YOLOPAFPN_focal(in_channels=fpn_in_channles,
                                       out_channels=in_channels,
                                       act=self.act,
                                       in_features=(1, 2, 3),
                                       depths=[2, 2, 18, 2],
                                       focal_levels=[4, 4, 4, 4],
                                       focal_windows=[3, 3, 3, 3],
                                       use_conv_embed=True,
                                       use_postln=True,
                                       use_postln_in_modulation=False,
                                       use_layerscale=True,
                                       base_dim=192,  # int(in_channels[0])
                                       depth=self.depth,
                                       width=self.width
                                       )


        else:
            raise NotImplementedError('backbone not support')
        from yolox.models.v_plus_head import YOLOVHead
        from yolox.models.yolov_plus import YOLOV

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03


        for layer in backbone.parameters():
            layer.requires_grad = False  # fix the backbone

        more_args = {'use_ffn': self.use_ffn, 'use_time_emd': self.use_time_emd, 'use_loc_emd': self.use_loc_emd,
                     'loc_fuse_type': self.loc_fuse_type, 'use_qkv': self.use_qkv,
                     'local_mask': self.local_mask, 'local_mask_branch': self.local_mask_branch,
                     'pure_pos_emb':self.pure_pos_emb,'loc_conf':self.loc_conf,'iou_base':self.iou_base,
                     'reconf':self.reconf,'ota_mode':self.ota_mode,'ota_cls':self.ota_cls,'traj_linking':self.traj_linking,
                     'iou_window':self.iou_window,'globalBlocks':self.globalBlocks,'use_pre_nms':self.use_pre_nms,
                     'cat_ota_fg':self.cat_ota_fg, 'agg_type':self.agg_type,'minimal_limit':self.minimal_limit,
                     'decouple_reg':self.decouple_reg,
                     }
        head = YOLOVHead(self.num_classes, self.width, in_channels=in_channels, heads=self.head, drop=self.drop_rate,
                         use_score=self.use_score, defualt_p=self.defualt_p, sim_thresh=self.sim_thresh,
                         pre_nms=self.pre_nms, ave=self.ave, defulat_pre=self.defualt_pre, test_conf=self.test_conf,
                         use_mask=self.use_mask,gmode=self.gmode,lmode=self.lmode,both_mode=self.both_mode,
                         localBlocks = self.localBlocks,**more_args)
        for layer in head.stems.parameters():
            layer.requires_grad = False  # set stem fixed
        for layer in head.reg_convs.parameters():
            layer.requires_grad = False
            layer.requires_grad = False
        for layer in head.cls_convs.parameters():
            layer.requires_grad = False
        for layer in head.reg_preds.parameters():
            layer.requires_grad = False

        self.model = YOLOV(backbone, head)

        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.model.apply(init_yolo)
        if self.fix_bn:
            self.model.apply(fix_bn)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2, pg3 = [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    if "head.stem" in k or "head.reg_convs" in k or "head.cls_convs" in k:
                        pg3.append(v.weight)
                        logger.info("head.weight: {}".format(k))
                    else:
                        pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group(
                {"params": pg3, "lr": lr * self.stem_lr_ratio, "weight_decay": self.weight_decay}
            )
            self.optimizer = optimizer

        return self.optimizer

