#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import copy
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets.vid_classes import VID_classes
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import random
import numpy as np
import pickle
from random import shuffle
import random
from yolox.models.post_process import online_previous_selection

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    # parser.add_argument(
    #     "demo", default="video", help="demo type, eg. image, video and webcam"
    # )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument('--save_result', default=True)

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='', type=str,
                        help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--dataset",
        default='vid',
        type=str,
        help="Decide pred classes"
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=576, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        self.model.half()
        self.tensor_type = torch.cuda.HalfTensor

    def inference(self, img, other_result=[]):

        if self.device == "gpu":
            img = img.type(self.tensor_type)
            img = img.cuda()

        with torch.no_grad():
            # stime = time.time()
            outputs, res_dic = self.model(img, other_result)
            # print('infer time:',time.time()-stime)
        return outputs, res_dic

    def to_repp(self, output, ratio, img_size, image_id):

        # preprocessing: resize
        if output == None:
            return []
        output[:, 0:4] /= ratio
        # cls = output[:, 6]
        # scores = output[:, 4] * output[:, 5]
        ih, iw = img_size
        width_diff = max(0, (ih - iw) // 2)
        height_diff = max(0, (iw - ih) // 2)
        pred_res = []
        output = output.cpu()
        output = output.numpy()
        for out in output:
            x_min, y_min, x_max, y_max = out[:4]
            y_min, x_min = max(0, y_min), max(0, x_min)
            y_max, x_max = min(img_size[0], y_max), min(img_size[1], x_max)
            width, height = x_max - x_min, y_max - y_min
            if width <= 0 or height <= 0: continue
            bbox_center = [(x_min + width_diff + width / 2) / max(iw, ih),
                           (y_min + height_diff + height / 2) / max(iw, ih)]
            pred = {'image_id': image_id, 'bbox': [x_min, y_min, width, height], 'bbox_center': bbox_center}
            pred['scores'] = out[-30:] * out[4]
            pred_res.append(pred)
        return pred_res

    def to_repp_heavy(self, output, ratio, img_size, image_id):
        if output == None:
            return []
        # preprocessing: resize
        output[:, 0:4] /= ratio
        # output[:,-30:] = output[:,-30:]#.sigmoid()
        # cls = output[:, 6]
        # scores = output[:, 4] * output[:, 5]
        ih, iw = img_size
        width_diff = max(0, (ih - iw) // 2)
        height_diff = max(0, (iw - ih) // 2)
        pred_res = []
        output = output.cpu()
        output = output.numpy()
        for out in output:
            x_min, y_min, x_max, y_max = out[:4]
            y_min, x_min = max(0, y_min), max(0, x_min)
            y_max, x_max = min(img_size[0], y_max), min(img_size[1], x_max)
            width, height = x_max - x_min, y_max - y_min
            if width <= 0 or height <= 0: continue
            bbox_center = [(x_min + width_diff + width / 2) / max(iw, ih),
                           (y_min + height_diff + height / 2) / max(iw, ih)]
            pred = {'image_id': image_id, 'bbox': [x_min, y_min, width, height], 'bbox_center': bbox_center}
            pred['scores'] = out[4:7]  # *out[4]
            pred_res.append(pred)
        return pred_res


def imageflow_demo(predictor, val_path, args):
    res = []
    dataset = np.load(val_path, allow_pickle=True).tolist()
    for element in dataset:
        res.append(element)

    repp_res = []
    file_writter = open(args.output_dir, 'wb')
    import tqdm
    progress_bar = tqdm
    cur_iter = 0
    for ele in res:
        cur_iter += 1
        if cur_iter % 10 == 0:
            print(str(cur_iter) + '/' + str(len(res)))
        # videos
        first_frame = ele[0]
        video_name = first_frame[first_frame.find('val'):first_frame.rfind('/')]

        preds_video = {}
        pred_res = []
        tmp_bank = [[], [], [], [], [], []]
        local_bank = [[], [], [], []]
        for frames in ele:
            tmp_imgs = []
            img = cv2.imread(os.path.join(args.data_dir,frames))
            height, width = img.shape[:2]
            ratio = min(predictor.test_size[0] / img.shape[0], predictor.test_size[1] / img.shape[1])
            img, _ = predictor.preproc(img, None, predictor.test_size)
            img = torch.from_numpy(img)
            tmp_imgs.append(img)
            imgs = torch.stack(tmp_imgs)
            other_result = online_previous_selection(tmp_bank, local_bank=local_bank, local=True)
            pred_result, res_dict = predictor.inference(imgs, other_result)
            N = int(res_dict['cls_scores'].shape[0] / len(tmp_imgs))
            for i in range(len(tmp_imgs)):
                tmp_bank[0].append(res_dict['cls_feature'][0, N * i:N * (i + 1)])
                tmp_bank[1].append(res_dict['reg_feature'][0, N * i:N * (i + 1)])
                tmp_bank[2].append(res_dict['cls_scores'][N * i:N * (i + 1)])
                tmp_bank[3].append(res_dict['reg_scores'][N * i:N * (i + 1)])
                if res_dict['msa'] != None:
                    local_bank[0].append(res_dict['msa'][N * i:N * (i + 1)])
                    local_bank[1].append(res_dict['boxes'][N * i:N * (i + 1)])
                    local_bank[2].append(res_dict['cls_scores'][N * i:N * (i + 1)])
                    local_bank[3].append(res_dict['reg_scores'][N * i:N * (i + 1)])
            for i in range(4):
                tmp_bank[i] = tmp_bank[i][-600:]
                local_bank[i] = local_bank[i][-600:]
            pred_res.append(pred_result[0])

        for pred, img_name in zip(pred_res, ele):
            point_idx = img_name.rfind('.')
            image_id = img_name[img_name.find('val'):point_idx]
            img_idx = img_name[img_name.rfind('/') + 1:point_idx]
            det_repp = predictor.to_repp_heavy(pred, ratio, [height, width], image_id)
            preds_video[img_idx] = det_repp
        # pickle.dump((video_name, preds_video), file_writter)
        repp_res.append([video_name, preds_video])
    for res in repp_res:
        pickle.dump((res[0], res[1]), file_writter)
    file_writter.close()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = exp.output_dir  # os.path.join(exp.output_dir, args.experiment_name)
    val_path = args.path
    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    args.data_dir = exp.data_dir
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    if args.dataset == 'vid':
        predictor = Predictor(model, exp, VID_classes, trt_file, decoder, args.device, args.legacy)

    imageflow_demo(predictor, val_path, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)


