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
from yolox.utils.dist import time_synchronized
from yolox.data.datasets import vid
from yolox.data.data_augment import Vid_Val_Transform
import random
import numpy as np
import pickle
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./yolox/data/datasets/val_seq.npy", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument('--save_result', default=True)

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./exps/yolov/yolov_s.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='./excluded/weights/yolov_s.pth', type=str, help="ckpt for eval")
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
    parser.add_argument("--conf", default=0.001, type=float, help="test conf")
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
    parser.add_argument('--output_dir', default='./YOLOX_outputs/yolov_s.pkl',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data_dir', default='', help= 'path to your dataset')
    parser.add_argument('--lframe', default=0,type=int, help='local frame num')
    parser.add_argument('--gframe', default=32,type=int, help='global frame num')
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


        #self.tensor_type = torch.cuda.FloatTensor

        self.model.half()
        self.tensor_type = torch.cuda.HalfTensor
    def inference(self, img,img_path=None,lframe=0,gframe=32):

        if self.device == "gpu":
            img = img.type(self.tensor_type)
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            infer_end = time_synchronized()
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=False
            )
            #print("inference time: {}".format(infer_end - t0))

        return outputs

    def to_repp(self, output,ratio, img_size,image_id):

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
            pred['scores'] = out[-30:]*out[4]
            pred_res.append(pred)
        return pred_res

    def to_repp_heavy(self, output,ratio, img_size,image_id):
        if output == None:
            return []
        # preprocessing: resize
        output[:, 0:4] /= ratio
        #output[:,-30:] = output[:,-30:]#.sigmoid()
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
            pred['scores'] = out[4:7]#*out[4]
            pred_res.append(pred)
        return pred_res


def gl_mode(predictor, val_path, args):
    import math
    lframe = args.lframe
    gframe = args.gframe
    repp_res = []
    dataset = np.load(val_path, allow_pickle=True).tolist()
    for element in dataset:
        frames = []
        outputs = []
        ori_frames = []
        preds_video = {}
        first_frame = element[0]
        video_name = first_frame[first_frame.find('val'):first_frame.rfind('/')]
        ## frame reading
        for frame in element:
            img = cv2.imread(frame)
            ori_frames.append(frame)
            frame, _ = predictor.preproc(frame, None, exp.test_size)
            frames.append(torch.tensor(frame))
        res = []
        ## frame dividing
        single_frame = None
        if len(frames) % lframe == 1:
            single_frame = frames[-1:]
            frames = frame[:-1]
        frame_len = len(frames)
        index_list = list(range(frame_len))
        # if args.gframe !=0 and args.lframe ==0:
        random.seed(42)
        random.shuffle(index_list)
        random.seed(42)
        random.shuffle(frames)
        split_num = int(frame_len / (gframe))  #
        for i in range(split_num):
            res.append(frames[i * gframe:(i + 1) * gframe])
        res.append(frames[(i + 1) * gframe:])

        ## frame inference
        features = []
        raw_pred = []
        for ele in res:
            if ele == []: continue
            ele = torch.stack(ele)
            t0 = time.time()
            feature, raw_pre = predictor.inference(ele, local_flag=False)
            features.append(feature)
            raw_pred += raw_pre

        features = torch.cat(features)
        # [j for _,j in sorted(zip(index_list,features))]
        raw_pred = [k for _, k in sorted(zip(index_list, raw_pred))]
        ori_idx = [k for _, k, in sorted(zip(index_list, list(range(frame_len))))]
        features = features[ori_idx]
        for i in range(int(math.ceil(frame_len / (lframe)))):
            ele = []
            ele.append(raw_pred[i * lframe:(i + 1) * lframe])
            ele.append(features[i * lframe:(i + 1) * lframe])
            res, res_ori = predictor.inference(ele, local_flag=True)
            outputs.extend(res)

        if single_frame != None:
            res_ori, res_ori = predictor.inference(single_frame, local_flag=False)
            outputs.extend(res_ori)

        ratio = min(predictor.test_size[0] / img.shape[0], predictor.test_size[1] / img.shape[1])
        height, width = img.shape[:2]
        for pred, img_name in zip(outputs, element):
            point_idx = img_name.rfind('.')
            image_id = img_name[img_name.find('val'):point_idx]
            img_idx = img_name[img_name.rfind('/') + 1:point_idx]
            det_repp = predictor.to_repp(pred, ratio, [height, width], image_id)
            preds_video[img_idx] = det_repp
        repp_res.append([video_name, preds_video])

    file_writter = open(args.output_dir, 'wb')
    for res in repp_res:
        pickle.dump((res[0], res[1]), file_writter)
    file_writter.close()


def imageflow_demo(predictor, val_path, args,exp):

    lframe = exp.lframe_val
    gframe = exp.gframe_val
    print(lframe,gframe)

    dataset_val = vid.VIDDataset(file_path='./yolox/data/datasets/val_seq.npy',
                                 img_size=exp.test_size, preproc=Vid_Val_Transform(),
                                 lframe=exp.lframe_val,
                                 gframe=exp.gframe_val,
                                 val=True,
                                 dataset_pth='/mnt/weka/scratch/yuheng.shi/dataset/VID',
                                 tnum=-1, formal=True)
    eval_loader = vid.vid_val_loader(batch_size=lframe+gframe,
                                    data_num_workers=12,
                                    dataset=dataset_val, )
    from tqdm import tqdm

    res_dict = {}
    for cur_iter, (imgs, _, info_imgs, label, path, time_embedding) in enumerate(
            tqdm(eval_loader)
    ):

        pred_res = predictor.inference(imgs)
        video_name = path[0][path[0].find('val'):path[0].rfind('/')]
        if video_name not in res_dict: res_dict[video_name] = {}
        for pred, img_name,info_img in zip(pred_res, path,info_imgs):
            point_idx = img_name.rfind('.')
            image_id = img_name[img_name.find('val'):point_idx]
            img_idx = img_name[img_name.rfind('/') + 1:point_idx]
            if img_idx in res_dict[video_name]: continue
            ratio = min(predictor.test_size[0] / info_img[0], predictor.test_size[1] / info_img[1])
            det_repp = predictor.to_repp_heavy(pred, ratio, [info_img[0], info_img[1]], image_id)
            res_dict[video_name][img_idx] = det_repp

    file_writter = open(args.output_dir, 'wb')
    for key,val in res_dict.items():
        pickle.dump((key, val), file_writter)
    file_writter.close()
    print('save result to {}'.format(args.output_dir))


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = exp.output_dir#os.path.join(exp.output_dir, args.experiment_name)
    val_path = args.path
    val_dataset = np.load(val_path, allow_pickle=True).tolist()

    vis_folder = None

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
        predictor = Predictor(model, exp, VID_classes, trt_file, decoder, args.device, args.legacy)

    current_time = time.localtime()
    imageflow_demo(predictor, val_path, args,exp)

import torch.backends.cudnn as cudnn
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    cudnn.benchmark = True
    exp.lframe_val = int(args.lframe)
    exp.gframe_val = int(args.gframe)

    main(exp, args)

