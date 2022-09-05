import os
import torch.nn as nn
from yolox.exp import Exp as MyExp
import torch.distributed as dist
import torch
import random
from yolox.data.datasets import vid

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path

        self.num_classes = 25
        self.data_dir = "/opt/dataset/OVIS"
        self.train_ann = "ovis_train.json"
        self.val_ann = "ovis_train.json"
        self.max_epoch = 7
        self.pre_no_aug = 0
        self.no_aug_epochs = 2
        self.warmup_epochs = 1
        self.eval_interval = 1
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.001 / 64.0
        self.input_size = (640, 960)
        self.test_size = (640, 960)
        self.test_conf = 0.001
        self.perspective = 0.0

        self.nmsthre = 0.5
        self.drop_rate = 0
        #COCO API has been changed

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
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, heads=4, drop=self.drop_rate,defualt_p=75,pre_nms=750)
        for layer in head.stems.parameters():
            layer.requires_grad = False  # set stem fixed
        for layer in head.reg_convs.parameters():
            layer.requires_grad = False
        for layer in head.reg_preds.parameters():
            layer.requires_grad = False
        for layer in head.cls_convs.parameters():
            layer.requires_grad = False
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

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import TrainTransform
        dataset = vid.OVIS(
                                img_size=self.input_size,
                                preproc=TrainTransform(
                                    max_labels=50,
                                    flip_prob=self.flip_prob,
                                    hsv_prob=self.hsv_prob),
                                mode='random',
                                lframe=0,
                                gframe=batch_size,
                                data_dir=self.data_dir,
                                name='train',
                                COCO_anno=os.path.join(self.data_dir, self.train_ann))

        dataset = vid.get_trans_loader(batch_size=batch_size, data_num_workers=4, dataset=dataset)
        return dataset


    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)

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
