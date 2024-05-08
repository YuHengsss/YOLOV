import os
from exps.yolov.yolov_base import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1 # 1#0.67
        self.width = 1  # 1#0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.max_epoch = 10
        self.no_aug_epochs = 2
        self.pre_no_aug = 2
        self.warmup_epochs = 1
        self.eval_interval = 1
        # COCO API has been changed








