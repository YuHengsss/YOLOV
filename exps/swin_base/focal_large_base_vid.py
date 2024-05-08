import os
import torch
import torch.nn as nn
from yolox.exp import Exp as MyExp
from loguru import logger


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = ''
        self.train_ann = "ILSVRC_FGFA_COCO.json"
        # name of annotation file for evaluation
        self.val_ann = "vid_val10000_coco.json"
        self.train_name = ''
        self.val_name = ''
        self.basic_lr_per_img = 0.00001 / 64.0
        self.save_history_ckpt = False
        self.max_epoch = 15
        self.input_size = (576,576)
        self.test_size = (576,576)
        self.eval_interval = 1
        self.warmup_epochs = 0
        self.no_aug_epochs = 15
        self.num_classes = 30
        self.test_conf = 0.001
        self.scheduler = 'warmcos'
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.focal_level = 4
        self.focal_windows = 3
        self.debug = True

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [96 * 4, 96 * 8, 96 * 16]
        out_channels = in_channels #[96 * 2, 96 * 4, 96 * 4, 96 * 8] #in_channels #[256, 512, 1024]
        from yolox.models import YOLOX, YOLOPAFPN_focal, YOLOXHead
        backbone = YOLOPAFPN_focal(in_channels=in_channels,  
                                  out_channels=out_channels,
                                  act=self.act,
                                  in_features=(1, 2, 3),
                                  depths=[2, 2, 18, 2],
                                  focal_levels = [4, 4, 4, 4],
                                  focal_windows = [3, 3, 3, 3],
                                  use_conv_embed = True, 
                                  use_postln = True,
                                  use_postln_in_modulation = False,
                                  use_layerscale = True,
                                  base_dim=192#int(in_channels[0])
                                  )
        head = YOLOXHead(self.num_classes, self.width,
                         in_channels=[96 * 4, 96 * 8, 96 * 16],
                         act=self.act, strides=[8, 16, 32],debug=self.debug)

        # for layer in backbone.backbone.parameters():
        #     layer.requires_grad = False  # fix the backbone
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2, pg3, pg4 = [], [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, 'absolute_pos_embed') or hasattr(v, 'relative_position_bias_table') or hasattr(v,
                                                                                                               'norm'):
                    if hasattr(v, 'weight'):
                        pg3.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    if "backbone.backbone" in k:  # check if it is the backbone
                        logger.info("backbone weight: {}".format(k))
                        pg4.append(v.weight)
                    else:
                        pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.AdamW(params=pg0, lr=lr, weight_decay=self.weight_decay)
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group(
                {"params": pg3, "weight_decay": 0}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group({"params": pg4, "weight_decay": self.weight_decay})


            self.optimizer = optimizer

        return self.optimizer




