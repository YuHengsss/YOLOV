#!/usr/bin/env python3
# -*- coding:utf-8 -*-


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
from yolox.utils import fuse_model, get_model_info, vis
from yolox.models.post_process import postprocess,get_linking_mat,post_linking
import copy
import random
import numpy as np
import pickle
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOV Testing")

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
        self.model.half()
        self.tensor_type = torch.cuda.HalfTensor
        if hasattr(exp,'traj_linking'):
            self.traj_linking = exp.traj_linking
        else:
            self.traj_linking = False
        self.post = post
    def inference(self, img,img_path=None,lframe=0,gframe=32):

        if self.device == "gpu":
            img = img.type(self.tensor_type)
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            if not self.traj_linking:
                outputs,outputs_ori = self.model(img,lframe=lframe,gframe=gframe)
                if len(outputs) <= 4: outputs = outputs_ori
            else:
                pred_result, adj_list, fc_output = self.model(img,lframe=lframe,gframe=gframe)
                outputs = [pred_result, adj_list, fc_output]
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
    Cls = exp.num_classes
    traj_linking = hasattr(exp, 'traj_linking') and exp.traj_linking
    print(lframe,gframe)
    eval_loader = exp.get_eval_loader(batch_size=gframe+lframe,tnum=-1,data_num_workers=12,formal=True,)
    from tqdm import tqdm

    res_dict = {}
    def add_res_dict(result,names,last_vid_name,info_imgs):
        #print('add {} frames to {}'.format(len(result),last_vid_name))
        for i in range(len(result)):
            img_name, pred = names[i], result[i]
            point_idx = img_name.rfind('.')
            image_id = img_name[img_name.find('val'):point_idx]
            img_idx = img_name[img_name.rfind('/') + 1:point_idx]
            if img_idx in res_dict[last_vid_name]: continue
            ratio = min(predictor.test_size[0] / info_imgs[0][0], predictor.test_size[1] / info_imgs[0][1])
            det_repp = predictor.to_repp_heavy(pred, ratio, [info_imgs[0][0], info_imgs[0][1]], image_id)
            res_dict[last_vid_name][img_idx] = det_repp

    pred_results, adj_lists, fc_outputs, names = [], [], [], []
    for cur_iter, (imgs, _, info_imgs, label, path, time_embedding) in enumerate(
            tqdm(eval_loader)
    ):
        video_name = path[0][path[0].find('val'):path[0].rfind('/')]
        # if video_name not in res_dict and traj_linking and len(pred_results): #Post process
        #     result = post_linking(fc_outputs, adj_lists, pred_results, P, Cls, names, exp)
        #     last_vid_name = names[0][names[0].find('val'):names[0].rfind('/')]
        #     add_res_dict(result,names,last_vid_name,last_info_imgs)
        #     pred_results, adj_lists, fc_outputs,names = [], [], [],[]

        if exp.lmode:
            if traj_linking:
                pred_result, adj_list, fc_output = predictor.inference(imgs,lframe=len(path),gframe=0)
                if len(pred_results) != 0: #skip the connection frame
                    pred_result = pred_result[1:]
                    fc_output = fc_output[1:]
                    path = path[1:]
                else:
                    res_dict[video_name] = {}
                pred_results.extend(pred_result)
                adj_lists.extend(adj_list)
                fc_outputs.append(fc_output)
                names.extend(path)
                last_info_imgs = info_imgs
                continue
            else:
                pred_res = predictor.inference(imgs,lframe=len(path),gframe=0)
        elif exp.gmode:
            pred_res = predictor.inference(imgs, lframe=0, gframe=len(path))

        video_name = path[0][path[0].find('val'):path[0].rfind('/')]
        if video_name not in res_dict: res_dict[video_name] = {}
        add_res_dict(pred_res,path,video_name,info_imgs)

    # if traj_linking:
    #     result = post_linking(fc_outputs, adj_lists, pred_results, P, Cls, names, exp)
    #     last_vid_name = names[0][names[0].find('val'):names[0].rfind('/')]
    #     add_res_dict(result,names,last_vid_name,last_info_imgs)

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


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    #exp.traj_linking = True and exp.lmode
    exp.lframe_val = int(args.lframe)
    exp.gframe_val = int(args.gframe)

    main(exp, args)

