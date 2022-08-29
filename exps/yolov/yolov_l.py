import os
import torch.nn as nn
from yolox.exp import Exp as MyExp
from yolox.data.datasets import vid
from loguru import logger
import torch


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1 # 1#0.67
        self.width = 1  # 1#0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path

        self.num_classes = 30
        self.data_dir = "path to your root dir"
        self.max_epoch = 10
        self.no_aug_epochs = 2
        self.pre_no_aug = 2
        self.warmup_epochs = 1
        self.eval_interval = 1
        self.min_lr_ratio = 0.5
        self.basic_lr_per_img = 0.002 / 64.0
        self.test_size = (512, 512)
        self.input_size = (512, 512)
        self.perspective = 0.0
        self.drop_rate = 0.0
        # COCO API has been changed

    def get_model(self):
        from yolox.models import YOLOPAFPN
        from yolox.models.yolovp_msa import YOLOXHead
        # from yolox.models.mhead_trans import YOLOXHead

        from yolox.models.myolox import YOLOX

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        # if getattr(self, "model", None) is None:
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        for layer in backbone.parameters():
            layer.requires_grad = False  # fix the backbone
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, heads=4, drop=self.drop_rate)
        for layer in head.stems.parameters():
            layer.requires_grad = False  # set stem fixed
        for layer in head.reg_convs.parameters():
            layer.requires_grad = False
        for layer in head.reg_preds.parameters():
            layer.requires_grad = False
        # for layer in head.obj_preds.parameters():
        #     layer.requires_grad = False
        for layer in head.cls_convs.parameters():
            layer.requires_grad = False
        self.model = YOLOX(backbone, head)
        #
        # def fix_bn(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         m.eval()
        self.model.apply(init_yolo)
        # self.model.apply(fix_bn)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter) and v.bias.requires_grad:
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter) and v.weight.requires_grad:
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_evaluator(self, val_loader):
        from yolox.evaluators.vid_evaluator_v2 import VIDEvaluator

        # val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VIDEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import TrainTransform
        from yolox.data.datasets.mosaicdetection import MosaicDetection_VID

        dataset = vid.VIDDataset(file_path='./yolox/data/datasets/train_seq.npy',
                                 img_size=self.input_size,
                                 preproc=TrainTransform(
                                     max_labels=50,
                                     flip_prob=self.flip_prob,
                                     hsv_prob=self.hsv_prob),
                                 lframe=0,  # batch_size,
                                 gframe=batch_size,
                                 dataset_pth=self.data_dir)

        dataset = vid.get_trans_loader(batch_size=batch_size, data_num_workers=4, dataset=dataset)
        return dataset

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs= 2*self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler
