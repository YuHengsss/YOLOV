#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets.vid_classes import VID_classes
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, time_synchronized
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
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='', type=str, help="ckpt for eval")
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
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
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
    parser.add_argument('--lframe', default=0, help='local frame num')
    parser.add_argument('--gframe', default=32, help='global frame num')
    parser.add_argument('--save_result', default=True)
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
        self.model = model.half()
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        self.tensor_type = torch.cuda.HalfTensor

    def inference(self, img, other_result=[]):

        if self.device == "gpu":
            img = img.type(self.tensor_type)
            img = img.cuda()

        with torch.no_grad():
            stime = time.time()
            outputs, res_dic = self.model(img, other_result, nms_thresh=self.nmsthre)
            infer_end = time_synchronized()
            print('infer time:', infer_end - stime)
        return outputs, res_dic

    def visual(self, output, img, ratio, cls_conf=0.0):
        if output is None:
            return img
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name, [image_name])
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    lframe = args.lframe
    gframe = args.gframe
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    os.makedirs(save_folder, exist_ok=True)
    ratio = min(predictor.test_size[0] / height, predictor.test_size[1] / width)
    save_path = os.path.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    frames = []
    outputs = []
    ori_frames = []
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            ori_frames.append(frame)
            frame, _ = predictor.preproc(frame, None, exp.test_size)
            frames.append(torch.tensor(frame))
        else:
            break

    res = []
    frame_len = len(frames)
    index_list = list(range(frame_len))
    tmp_bank = [[], [], [], []]
    local_bank = [[], [], [], []]
    for frame_num, frame in enumerate(frames):
        tmp_imgs = []

        img = frame
        tmp_imgs.append(img)
        # if frame_num == 0:
        #     tmp_imgs = tmp_imgs + frames[-31:]
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
        outputs.extend(pred_result)

    for output, img in zip(outputs, ori_frames[:len(outputs)]):

        result_frame = predictor.visual(output, img, ratio, cls_conf=args.conf)
        if args.save_result:
            vid_writer.write(result_frame)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

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
    else:
        predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.legacy)
    current_time = time.localtime()

    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
