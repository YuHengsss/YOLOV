import torch
import torch.nn as nn
import os
import random
from yolox.exp import Exp as MyExp
import torch.distributed as dist

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 30
        self.data_dir = "/home/xteam/yh/YOLOV/"
        self.train_ann = "ILSVRC_final_coco.json"
        self.val_ann = "annotations_val10000_coco.json"
        self.max_epoch = 7
        self.no_aug_epochs = 2
        self.warmup_epochs = 1
        self.eval_interval = 1
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.005 / 64.0
        self.input_size = (704, 704)
        self.test_size = (704, 704)
        self.multiscale_range = 4
        self.test_conf = 0.001
        self.nmsthre = 0.5
        self.width = 1

    def get_model(self):
        from yolox.models.backbones.ELANNet import ELANNet,ELANFPNP6
        from yolox.models.yolo_head import YOLOXHead
        from yolox.models.yolov7 import YOLOv7
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 768, 1024] # be the same with fpn output channel
        backbone = ELANNet(arch='W6',return_idx=[2, 3, 4, 5])
        fpn = ELANFPNP6(arch='W6',use_aux=False)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,strides=[8, 16, 32, 64])

        self.model = YOLOv7(backbone,fpn,head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 64) - self.multiscale_range
                max_size = int(self.input_size[0] / 64) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(64 * size), 64 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size