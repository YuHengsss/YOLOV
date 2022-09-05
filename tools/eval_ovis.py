#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import json
import os
import time

import numpy
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import matplotlib.pyplot as plt

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/opt/dataset/OVIS/annotations_valid.json", help="path to image name list"
    )  # /opt/dataset/EPIC-KITCHENS/P32/object_detection_images/P32_10
    #
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='../exps/yolo_ovis/yoloxx_ovis.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='../YOLOX_outputs/yoloxx_ovis/epoch_9_ckpt.pth', type=str,
                        help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.001, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
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
            cls_names=None,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
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
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        # img_show = img.transpose((1,2,0)).astype(numpy.uint8)
        # plt.figure()
        # plt.imshow(img_show)
        # plt.show()
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def batch_inference(self, img):
        width, height = img.shape[-2:]
        ratio = min(self.test_size[0] / 1080, self.test_size[1] / 1920)
        # img, _ = self.preproc(img, None, self.test_size)

        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        result = []
        for ele in outputs:
            if ele != None:
                ele[:, :4] /= ratio
                result.append(ele)
            else:
                result.append(torch.Tensor([[0, 0, 0, 0, 0, 0, 0]]).to('cuda'))
        return result

    def convert_and_visual(self, output, img_info, cls_conf=0.35, vis_flag=False):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        if vis_flag:
            vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names, 1.0)
            return vis_res
        else:
            return bboxes, cls, scores


def image_demo(predictor, vis_folder, path, current_time, save_result, img_flag=False,
               dataset_dir='/opt/dataset/OVIS/valid/'):
    # b1 inference
    files = list(path.keys())
    #files.sort()
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    result_list = []
    from tqdm import tqdm
    progress_bar = tqdm
    for ele in progress_bar(files):
        image_name = os.path.join(dataset_dir, ele)
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.convert_and_visual(outputs[0], img_info, 0.05, vis_flag=img_flag)

        # visualize
        if img_flag:
            if save_result:
                os.makedirs(save_folder, exist_ok=True)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                logger.info("Saving detection result in {}".format(save_file_name))
                cv2.imwrite(save_file_name, result_image)
        else:
            bboxes, cls, scores = result_image
            for i in range(bboxes.shape[0]):
                box = bboxes[i]
                x = int(min(max(int(box[0]),1),img_info["width"]-1))
                y = int(min(max(int(box[1]),1),img_info["height"]-1))
                x1 =  int(min(max(int(box[2]),1),img_info["width"]-1))
                y1 =  int(min(max(int(box[3]),1),img_info["height"]-1))

                width = x1 - x
                height = y1 - y
                if width < 1 or height < 1:
                    continue
                category_id = int(cls[i]) + 1
                score = float(scores[i])
                tmp_dic = {
                    "image_id":  path[ele] ,
                    "category_id": category_id,
                    "bbox": [x, y, width, height],
                    "score": score,
                }
                result_list.append(tmp_dic)
    if not img_flag:
        save_name = 'ovis_valid.json'
        out_file = open(save_name, "w")
        json.dump(result_list, out_file)
        out_file.close()
        print('file saves:' + save_name)
    return None


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
        if args.fp16:
            model.half()  # to FP16
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

    names, name_id_dic = parse_anno(args.path)
    predictor = Predictor(
        model, exp, names, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, name_id_dic, current_time, args.save_result, img_flag=True)
    elif args.demo == "eval":
        image_demo(predictor, vis_folder, name_id_dic, current_time, args.save_result)


def parse_anno(path):
    with open(path, 'r') as ovis:
        ovis = json.load(ovis)
    name_list = []
    name_id_dic = {}
    for name in ovis['categories']:
        name_list.append(name['name'])
    for video in ovis['videos']:
        for fid, frame in enumerate(video['file_names']):
            name_id_dic[frame] = str(video['id']) + '_' + str(fid + 1)

    return name_list, name_id_dic

def covert_format(path = 'ovis_valid.json'):
    with open(path, 'r') as ovis:
        ovis = json.load(ovis)
    result = []
    for ele in ovis:

        ele['category_id'] = ele['category_id'] + 1
        result.append(ele)
    save_name = path
    out_file = open(save_name, "w")
    json.dump(result, out_file)
    out_file.close()
    print('file saves:' + save_name)



if __name__ == "__main__":

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

# python tools/demo.py video -f yolox/exp/yoloxx_bn.py -c YOLOX_outputs/yoloxx_bn/latest_ckpt.pth --path /media/ssd/ILSVRC2015/Data/VID/snippets/test/ILSVRC2015_test_00125000.mp4 --conf 0.1 --nms 0.5  --save_result --device gpu --fp16
