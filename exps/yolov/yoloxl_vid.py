import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1
        self.width = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path

        self.num_classes = 30
        self.data_dir = "path to your root dir"

        self.train_ann = "vid_train_coco.json"
        self.val_ann = "vid_val10000_coco.json"
        self.max_epoch = 7
        self.no_aug_epochs = 1
        self.warmup_epochs = 0
        self.eval_interval = 1
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.005 / 64.0
        self.input_size = (512, 512)
        self.test_size = (576,576)
        self.test_conf = 0.001
        self.nmsthre = 0.5
        #COCO API has been changed
