import torch
import torch.nn as nn
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 30
        self.data_dir = "/home/hdr/yh/YOLOV/"
        self.train_ann = "ILSVRC_final_coco.json"
        self.val_ann = "annotations_val10000_coco.json"
        self.max_epoch = 8
        self.no_aug_epochs = 2
        self.warmup_epochs = 1
        self.eval_interval = 1
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.003 / 64.0
        self.input_size = (512, 512)
        self.test_size = (576, 576)
        self.test_conf = 0.001
        self.nmsthre = 0.5
        self.width = 1.25

    def get_model(self):
        from yolox.models.backbones.ELANNet import ELANNet,ELANFPN
        from yolox.models.yolo_head import YOLOXHead
        from yolox.models.yolov7 import YOLOv7
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024] # be the same with fpn output channel
        backbone = ELANNet(arch='X')
        fpn = ELANFPN(arch='X',in_channels=[640,1280,640])
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)

        self.model = YOLOv7(backbone,fpn,head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model