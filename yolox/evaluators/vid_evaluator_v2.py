#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import copy
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
from yolox.evaluators.coco_evaluator import per_class_AR_table, per_class_AP_table
import torch
import pycocotools.coco

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

vid_classes = (
    'airplane', 'antelope', 'bear', 'bicycle',
    'bird', 'bus', 'car', 'cattle',
    'dog', 'domestic_cat', 'elephant', 'fox',
    'giant_panda', 'hamster', 'horse', 'lion',
    'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel',
    'tiger', 'train', 'turtle', 'watercraft',
    'whale', 'zebra'
)


# from yolox.data.datasets.vid_classes import Arg_classes as  vid_classes

class VIDEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
            self, dataloader, img_size, confthre, nmsthre,
            num_classes, testdev=False, gl_mode=False,
            lframe=0, gframe=32,**kwargs
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.id = 0
        self.box_id = 0
        self.id_ori = 0
        self.box_id_ori = 0
        self.gl_mode = gl_mode
        self.lframe = lframe
        self.gframe = gframe
        self.kwargs = kwargs
        self.vid_to_coco = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "airplane"},
                           {"supercategorie": "", "id": 1, "name": "antelope"},
                           {"supercategorie": "", "id": 2, "name": "bear"},
                           {"supercategorie": "", "id": 3, "name": "bicycle"},
                           {"supercategorie": "", "id": 4, "name": "bird"},
                           {"supercategorie": "", "id": 5, "name": "bus"},
                           {"supercategorie": "", "id": 6, "name": "car"},
                           {"supercategorie": "", "id": 7, "name": "cattle"},
                           {"supercategorie": "", "id": 8, "name": "dog"},
                           {"supercategorie": "", "id": 9, "name": "domestic_cat"},
                           {"supercategorie": "", "id": 10, "name": "elephant"},
                           {"supercategorie": "", "id": 11, "name": "fox"},
                           {"supercategorie": "", "id": 12, "name": "giant_panda"},
                           {"supercategorie": "", "id": 13, "name": "hamster"},
                           {"supercategorie": "", "id": 14, "name": "horse"},
                           {"supercategorie": "", "id": 15, "name": "lion"},
                           {"supercategorie": "", "id": 16, "name": "lizard"},
                           {"supercategorie": "", "id": 17, "name": "monkey"},
                           {"supercategorie": "", "id": 18, "name": "motorcycle"},
                           {"supercategorie": "", "id": 19, "name": "rabbit"},
                           {"supercategorie": "", "id": 20, "name": "red_panda"},
                           {"supercategorie": "", "id": 21, "name": "sheep"},
                           {"supercategorie": "", "id": 22, "name": "snake"},
                           {"supercategorie": "", "id": 23, "name": "squirrel"},
                           {"supercategorie": "", "id": 24, "name": "tiger"},
                           {"supercategorie": "", "id": 25, "name": "train"},
                           {"supercategorie": "", "id": 26, "name": "turtle"},
                           {"supercategorie": "", "id": 27, "name": "watercraft"},
                           {"supercategorie": "", "id": 28, "name": "whale"},
                           {"supercategorie": "", "id": 29, "name": "zebra"}],
            'images': [],
            'licenses': []
        }
        self.vid_to_coco_ori = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "airplane"},
                           {"supercategorie": "", "id": 1, "name": "antelope"},
                           {"supercategorie": "", "id": 2, "name": "bear"},
                           {"supercategorie": "", "id": 3, "name": "bicycle"},
                           {"supercategorie": "", "id": 4, "name": "bird"},
                           {"supercategorie": "", "id": 5, "name": "bus"},
                           {"supercategorie": "", "id": 6, "name": "car"},
                           {"supercategorie": "", "id": 7, "name": "cattle"},
                           {"supercategorie": "", "id": 8, "name": "dog"},
                           {"supercategorie": "", "id": 9, "name": "domestic_cat"},
                           {"supercategorie": "", "id": 10, "name": "elephant"},
                           {"supercategorie": "", "id": 11, "name": "fox"},
                           {"supercategorie": "", "id": 12, "name": "giant_panda"},
                           {"supercategorie": "", "id": 13, "name": "hamster"},
                           {"supercategorie": "", "id": 14, "name": "horse"},
                           {"supercategorie": "", "id": 15, "name": "lion"},
                           {"supercategorie": "", "id": 16, "name": "lizard"},
                           {"supercategorie": "", "id": 17, "name": "monkey"},
                           {"supercategorie": "", "id": 18, "name": "motorcycle"},
                           {"supercategorie": "", "id": 19, "name": "rabbit"},
                           {"supercategorie": "", "id": 20, "name": "red_panda"},
                           {"supercategorie": "", "id": 21, "name": "sheep"},
                           {"supercategorie": "", "id": 22, "name": "snake"},
                           {"supercategorie": "", "id": 23, "name": "squirrel"},
                           {"supercategorie": "", "id": 24, "name": "tiger"},
                           {"supercategorie": "", "id": 25, "name": "train"},
                           {"supercategorie": "", "id": 26, "name": "turtle"},
                           {"supercategorie": "", "id": 27, "name": "watercraft"},
                           {"supercategorie": "", "id": 28, "name": "whale"},
                           {"supercategorie": "", "id": 29, "name": "zebra"}],
            'images': [],
            'licenses': []
        }
        self.testdev = testdev
        self.tmp_name_ori = './ori_pred.json'
        self.tmp_name_refined = './refined_pred.json'
        self.gt_ori = './gt_ori.json'
        self.gt_refined = './gt_refined.json'

    def evaluate(
            self,
            model,
            distributed=False,
            half=True,
            trt_file=None,
            decoder=None,
            test_size=None,
            img_path=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        labels_list = []
        ori_data_list = []
        ori_label_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, _, info_imgs, label, path, time_embedding) in enumerate(
                progress_bar(self.dataloader)
        ):

            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                outputs, ori_res = model(imgs,
                                         lframe=self.lframe,
                                         gframe = self.gframe)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            if self.gl_mode:
                local_num = int(imgs.shape[0] / 2)
                info_imgs = info_imgs[:local_num]
                label = label[:local_num]
            if self.kwargs.get('first_only',False):
                info_imgs = [info_imgs[0]]
                label = [label[0]]
            temp_data_list, temp_label_list = self.convert_to_coco_format(outputs, info_imgs, copy.deepcopy(label))
            data_list.extend(temp_data_list)
            labels_list.extend(temp_label_list)

        self.vid_to_coco['annotations'].extend(labels_list)
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        del labels_list
        eval_results = self.evaluate_prediction(data_list, statistics)
        del data_list
        self.vid_to_coco['annotations'] = []

        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, labels):
        data_list = []
        label_list = []
        frame_now = 0

        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            # if frame_now>=self.lframe: break
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id = self.box_id + 1
                label_list.append(label_pred_data)
            self.vid_to_coco['images'].append({'id': self.id})

            if output is None:
                self.id = self.id + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]
            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
            self.id = self.id + 1
            frame_now = frame_now + 1

        return data_list, label_list

    def convert_to_coco_format_ori(self, outputs, info_imgs, labels):

        data_list = []
        label_list = []
        frame_now = 0
        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id_ori,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id_ori = self.box_id_ori + 1
                label_list.append(label_pred_data)

                # print('label:',label_pred_data)

            self.vid_to_coco_ori['images'].append({'id': self.id_ori})

            if output is None:
                self.id_ori = self.id_ori + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            # print(cls.shape)
            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

            self.id_ori = self.id_ori + 1
            frame_now = frame_now + 1
        return data_list, label_list

    def evaluate_prediction(self, data_dict, statistics, ori=False):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "NMS", "inference"],
                [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
            )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:

            _, tmp = tempfile.mkstemp()
            if ori:
                json.dump(self.vid_to_coco_ori, open(self.gt_ori, 'w'))
                json.dump(data_dict, open(self.tmp_name_ori, 'w'))
                json.dump(self.vid_to_coco_ori, open(tmp, "w"))
            else:
                json.dump(self.vid_to_coco, open(self.gt_refined, 'w'))
                json.dump(data_dict, open(self.tmp_name_refined, 'w'))
                json.dump(self.vid_to_coco, open(tmp, "w"))

            cocoGt = pycocotools.coco.COCO(tmp)
            # TODO: since pycocotools can't process dict in py36, write data to json file.

            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()

            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            info += "per class AP:\n" + AP_table + "\n"

            AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            info += "per class AR:\n" + AR_table + "\n"
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
