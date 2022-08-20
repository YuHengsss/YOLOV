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
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import random
import REPP
import json
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
        type = str,
        help = "Decide pred classes"
    )
    parser.add_argument("--conf", default=0.2, type=float, help="test conf")
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
    parser.add_argument('--repp_cfg',default='./tools/yolo_repp_cfg.json' ,help='repp cfg filename', type=str)
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
        post = None
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
        self.model = model.half()
        self.post = post

    def inference(self, img,img_path=None):
        tensor_type = torch.cuda.HalfTensor
        if self.device == "gpu":
            img = img.cuda()
            img = img.type(tensor_type)
        with torch.no_grad():
            t0 = time.time()
            outputs,outputs_ori = self.model(img)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs

    def visual(self, output,img,ratio, cls_conf=0.0):

        if output is None:
            return img
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

    def to_repp_heavy(self, output,ratio, img_size,image_id):
        if output == None:
            return []
        # preprocessing: resize
        output[:, 0:4] /= ratio
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
            pred['scores'] = out[4:7]#*out[4]
            pred_res.append(pred)
        return pred_res

    def convert_to_post(self,pred_res,ratio,img_size):
        res_dic = {}
        for idx, ele in enumerate(pred_res):
            res_dic[str(idx)] = self.to_repp_heavy(ele,ratio,img_size,str(idx))
        return res_dic
    def convert_to_ori(self,post_res,frame_num):
        tmp_list = []
        res_list = []
        for i in range(frame_num):
            tmp_list.append([])
        for ele in post_res:
            x1,y1,x2,y2 = ele['bbox'][0],ele['bbox'][1],ele['bbox'][0]+ele['bbox'][2],ele['bbox'][1]+ele['bbox'][3]
            tmp_list[int(ele['image_id'])].append(torch.Tensor([x1,y1,x2,y2,1,ele['score'],ele['category_id']]))
        for ele in tmp_list:
            if ele == []:
                res_list.append(None)
            else:
                res_list.append(torch.stack(ele))
        return res_list
def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name,[image_name])
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
    random.seed(42)
    random.shuffle(index_list)
    random.seed(42)
    random.shuffle(frames)

    split_num = int(frame_len / (gframe))#

    for i in range(split_num):
        res.append(frames[i * gframe:(i + 1) * gframe])
    res.append(frames[(i + 1) * gframe:])

    for ele in res:
        if ele == []: continue
        ele = torch.stack(ele)
        t0 = time.time()
        outputs.extend(predictor.inference(ele))

    outputs = [j for _,j in sorted(zip(index_list,outputs))]
    out_post_format = predictor.convert_to_post(outputs,ratio,[height,width])
    out_post = predictor.post(out_post_format)
    outputs_post = predictor.convert_to_ori(out_post,frame_len)
    for output,img in zip(outputs_post,ori_frames):
        result_frame = predictor.visual(output,img,1,cls_conf=args.conf)
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
    if args.dataset=='vid':
        repp_params = json.load(open(args.repp_cfg, 'r'))
        post = REPP.REPP(**repp_params)
        predictor = Predictor(model, exp, VID_classes, trt_file, decoder, args.device, args.legacy,post=post)
    else:
        predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.legacy)
    current_time = time.localtime()

    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
