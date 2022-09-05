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
        default='../exps/yolo_ovis/yolovs_ovis_100_100_750.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='../YOLOX_outputs/yolovs_ovis_100_100_750/latest_ckpt.pth', type=str,
                        help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--save_dir",
        default="ovis_valid_vid.json",
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
        self.tensor_type = torch.cuda.HalfTensor
    def inference(self, img, img_path=None):

        if self.device == "gpu":
            img = img.type(self.tensor_type)
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs, outputs_ori = self.model(img)
        if len(outputs) <= 4: outputs = outputs_ori

        return outputs

    def visual(self, output, img, ratio, cls_conf=0.0):

        if output is None:
            return img
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names,t_size=0.6)
        return vis_res
    def convert_and_visual(self, output, ratio):

        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        return bboxes, cls, scores

import random
def image_demo(predictor,path, dataset_dir='/opt/dataset/OVIS/valid/',vid_list = [],args = ''):
    # b1 inference
    lframe = 0
    gframe = 32
    res = []
    result_list = []
    for element in vid_list:
        res_video = []
        ele_len = len(element)
        if ele_len <= lframe + gframe:
            # TODO fix the unsolved part
            res_video.append(element)
        else:
            if lframe == 0:
                split_num = int(ele_len / (gframe))
                random.shuffle(element)
                for i in range(split_num):
                    res_video.append(element[i * gframe:(i + 1) * gframe])
                res_video.append(element[(i + 1) * gframe:])
            else:
                return None
        res.append(res_video)

    from tqdm import tqdm
    progress_bar = tqdm

    for ele in progress_bar(res):
        first_frame = ele[0][0]
        for frames in ele:
            #frames
            if frames == []: continue
            tmp_imgs = []
            h, w = 0,0
            for img in frames:
                img = cv2.imread(os.path.join(dataset_dir,img))
                h, w = img.shape[:2]
                img, _ = predictor.preproc(img, None, predictor.test_size)
                img = torch.from_numpy(img)
                tmp_imgs.append(img)
            imgs = torch.stack(tmp_imgs)
            pred_res = predictor.inference(imgs)
            ratio = min(predictor.test_size[0] / h, predictor.test_size[1] / w)

            for pred,img_name in zip(pred_res,frames):
                if type(pred) != type(None):
                    result_image = predictor.convert_and_visual(pred,ratio)
                    bboxes, cls, scores = result_image
                    for i in range(bboxes.shape[0]):
                        box = bboxes[i]
                        x = int(min(max(int(box[0]), 1), w-1))
                        y = int(min(max(int(box[1]), 1), h-1))
                        x1 = int(min(max(int(box[2]), 1), w-1))
                        y1 = int(min(max(int(box[3]), 1), h-1))
                        width = x1 - x
                        height = y1 - y
                        if width < 1 or height < 1:
                            continue
                        category_id = int(cls[i]) + 1
                        score = float(scores[i])
                        tmp_dic = {
                            "image_id": path[img_name],
                            "category_id": category_id,
                            "bbox": [x, y, width, height],
                            "score": score,
                        }
                        result_list.append(tmp_dic)

    save_name = args.save_dir
    out_file = open(save_name, "w")
    json.dump(result_list, out_file)
    out_file.close()
    print('file saves:' + save_name)
    return None

from new_demo import get_image_list
def image_visual(predictor,vis_folder,current_time, data_dir='/opt/dataset/OVIS/valid/'):
    gframe = 32
    files = get_image_list(data_dir)
    ori_frames = [cv2.imread(file) for file in files]
    frames = []

    for frame in ori_frames:
        frame, _ = predictor.preproc(frame, None, exp.test_size)
        frames.append(torch.tensor(frame))
    frame_len = len(frames)
    index_list = list(range(frame_len))
    random.seed(41)
    random.shuffle(index_list)
    random.seed(41)
    random.shuffle(frames)
    split_num = int(frame_len / (gframe))
    ratio = min(predictor.test_size[0] / ori_frames[0].shape[0], predictor.test_size[1] / ori_frames[0].shape[1])
    res = []
    outputs = []
    for i in range(split_num):
        res.append(frames[i * gframe:(i + 1) * gframe])
    res.append(frames[(i + 1) * gframe:])
    for ele in res:
        if ele == []: continue
        ele = torch.stack(ele)
        t0 = time.time()
        outputs.extend(predictor.inference(ele))
    outputs = [j for _, j in sorted(zip(index_list, outputs))]
    for idx,(output,img) in enumerate(zip(outputs,ori_frames[:len(outputs)])):
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        result_frame = predictor.visual(output,img,ratio,cls_conf=0.3)
        save_file_name = os.path.join(save_folder, os.path.basename(files[idx]))
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_frame)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

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

    names, name_id_dic,vid_list = parse_anno(args.path)


    predictor = Predictor(
        model, exp, names, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_visual(predictor,vis_folder,current_time, data_dir= '/opt/dataset/OVIS/valid/5834b092')
    elif args.demo == "eval":
        image_demo(predictor, name_id_dic,vid_list=vid_list,args = args)

    #'/opt/dataset/OVIS/valid/af48b2f9'  elephant - person case --rare pose  2160*1080
    # /opt/dataset/OVIS/test/2f421dd4   dog - person case  --occlusion Fail
    # /opt/dataset/OVIS/test/7abfd898 rabbit - person case 1920*1080 10/77/130
    # /opt/dataset/OVIS/test/9e50c4fd zibra case 3840*2160
    # /opt/dataset/OVIS/test/41bba0e2 monkey case 1104*545
    # /opt/dataset/OVIS/test/a5d3fe9c turtle case 1280*663 Fail
    # /opt/dataset/OVIS/test/e99f1386 sheep case 1920*1080
    # /opt/dataset/OVIS/valid/2ab06287 dog case 1680
    # /opt/dataset/OVIS/valid/6a6547d7 rabbit case 1104
    # /opt/dataset/OVIS/valid/5834b092 tiger bus case 960
def parse_anno(path):
    with open(path, 'r') as ovis:
        ovis = json.load(ovis)
    name_list = []
    name_id_dic = {}
    vid_list = []
    for name in ovis['categories']:
        name_list.append(name['name'])
    for video in ovis['videos']:
        vid_list.append(video['file_names'])
        for fid, frame in enumerate(video['file_names']):
            name_id_dic[frame] = str(video['id']) + '_' + str(fid + 1)

    return name_list, name_id_dic,vid_list



if __name__ == "__main__":

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)

# python tools/demo.py video -f yolox/exp/yoloxx_bn.py -c YOLOX_outputs/yoloxx_bn/latest_ckpt.pth --path /media/ssd/ILSVRC2015/Data/VID/snippets/test/ILSVRC2015_test_00125000.mp4 --conf 0.1 --nms 0.5  --save_result --device gpu --fp16
