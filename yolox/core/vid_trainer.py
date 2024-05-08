#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from yolox.data.datasets import vid

#from yolox.data import DataPrefetcher
from yolox.data.datasets.vid import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
import re

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:

        m.eval()

def extract_values(text):
    AP75 = re.search(r'Average Precision  \(AP\) @\[ IoU=0.75.*? \] = (\d+\.\d+)', text).group(1)
    try:
        AP_small = re.search(r'Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area= small.*? \] = (\d+\.\d+)', text).group(1)
        AP_medium = re.search(r'Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=medium.*? \] = (\d+\.\d+)', text).group(1)
        AP_large = re.search(r'Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area= large.*? \] = (\d+\.\d+)', text).group(1)
        AR1 = re.search(r'Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=  1 \] = (\d+\.\d+)', text).group(1)
        AR10 = re.search(r'Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets= 10 \] = (\d+\.\d+)', text).group(1)
        AR100 = re.search(r'Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = (\d+\.\d+)', text).group(1)
        AR_small = re.search(r'Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area= small.*? \] = (\d+\.\d+)', text).group(1)
        AR_medium = re.search(r'Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area=medium.*? \] = (\d+\.\d+)', text).group(1)
        AR_large = re.search(r'Average Recall     \(AR\) @\[ IoU=0.50:0.95 \| area= large.*? \] = (\d+\.\d+)', text).group(1)
    except Exception:
        AP_small = 0
        AP_medium = 0
        AP_large = 0
        AR1 = 0
        AR10 = 0
        AR100 = 0
        AR_small = 0
        AR_medium = 0
        AR_large = 0

    return {
        'AP75': float(AP75),
        'AP_small': float(AP_small),
        'AP_medium': float(AP_medium),
        'AP_large': float(AP_large),
        'AR1': float(AR1),
        'AR10': float(AR10),
        'AR100': float(AR100),
        'AR_small': float(AR_small),
        'AR_medium': float(AR_medium),
        'AR_large': float(AR_large)
    }


class Trainer:
    def __init__(self, exp, args, val_loader,val=False):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        #self.prefetcher = vid.DataPrefetcher(train_loader)
        #self.train_loader = train_loader
        self.val_loader = val_loader
        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)
        self.lr = 0
        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

        if val:
            self.evaluate()
            return

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets,_ = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size,
                                            )
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, lframe = self.exp.lframe,gframe = self.exp.gframe)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        self.max_iter = int(self.prefetcher.max_iter)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()


        self.evaluator = self.exp.get_evaluator(
            val_loader=self.val_loader
        )
        # Tensorboard logger
        if self.rank == 0:
            #if self.args.logger == "tensorboard":
            self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        # if (self.epoch + 1- self.exp.warmup_epochs - self.exp.pre_no_aug) % 4 ==0 \
        #         and (self.epoch + 1- self.exp.warmup_epochs - self.exp.pre_no_aug) \
        #         and (self.epoch + 1 < self.max_epoch - self.exp.no_aug_epochs):
        #     self.train_loader = self.exp.get_data_loader(
        #         batch_size=self.args.batch_size,
        #         is_distributed=self.is_distributed,
        #         no_aug=self.no_aug,
        #         cache_img=self.args.cache,
        #     )
        #     logger.info('Refreshing dataloader')
        if self.epoch + 1 >= self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.dataset.enable_mosaic = False
            self.prefetcher = vid.DataPrefetcher(self.train_loader)
            # if self.is_distributed:
            #     self.model.module.head.use_l1 = True
            # else:
            #     self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug and self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")
        elif 0< self.epoch + 1 <= self.exp.pre_no_aug + self.exp.warmup_epochs:
            self.train_loader.dataset.enable_mosaic = False
            self.prefetcher = vid.DataPrefetcher(self.train_loader)
        else:
            logger.info("--->Including mosaic aug now!")
            self.train_loader.dataset.enable_mosaic = True
            self.prefetcher = vid.DataPrefetcher(self.train_loader)


    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
        self.prefetcher = vid.DataPrefetcher(self.train_loader)

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                None, self.epoch, self.rank, self.is_distributed
            )
        if self.iter % 2000 ==0:
            self.save_ckpt(ckpt_name='latest')

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module
        self.evaluator = self.exp.get_evaluator(
            val_loader=self.val_loader
        )
        summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed,self.args.fp16
        )
        self.model.train()

        ap50_95 = summary[0]
        ap50 = summary[1]

        summary_info = summary[-1]
        detail_info = extract_values(summary_info)
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            for k, v in detail_info.items():
                self.tblogger.add_scalar("val/{}".format(k), v, self.epoch + 1)
            self.tblogger.add_scalar("lr", self.lr, self.epoch + 1)
            logger.info('\n'+ str(summary[-1]))

        synchronize()

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def evaluate(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        model = self.resume_train(model)
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)
        self.model = model
        evalmodel = self.model
        evalmodel.eval()
        self.evaluator = self.exp.get_evaluator(
            val_loader=self.val_loader
        )
        summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed, self.amp_training
        )
        self.model.train()
        if self.rank == 0:
            logger.info('\n'+ str(summary[-1]))
        synchronize()
        return None

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
