import contextlib
import io
import os
import json
import tempfile
from loguru import logger
from tqdm import tqdm
from xml.dom import minidom
import torch
import pycocotools.coco
import pickle
import argparse
from matplotlib import pyplot as plt
from utils import load_json
from yolox.evaluators.coco_evaluator import per_class_AP_table,per_class_AR_table
from yolox.utils import (
    is_main_process,
    xyxy2xywh
)
XML_EXT = [".xml"]
import numpy as np
def make_parser():
    parser = argparse.ArgumentParser("IMDB TO COCO!")
    parser.add_argument(
        "--path", default="./excluded/post/yolov_s.pkl", help="path to images or video"
    )
    parser.add_argument(
        "--dataset_dir", default="/mnt/weka/scratch/yuheng.shi/dataset/VID", help="path to images or video"
    )
    parser.add_argument(
        "--tide", default=False, action="store_true",
    )
    return parser


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

name_list = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']
numlist = range(30)
name_num = dict(zip(name_list,numlist))

def get_xml_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in XML_EXT:
                image_names.append(apath)


def get_video_frame_iterator(filename, from_python_2=False):
    with open(filename, 'rb') as f:
        while True:
            try:
                if from_python_2:
                    yield pickle.load(f, encoding='latin1')
                else:
                    yield pickle.load(f)
            except EOFError:
                return
            except Exception as e:
                print('Unable to load data ', filename, ':', e)
                raise ValueError('Unable to load data ', filename, ':', e)



class REPP_to_COCO_Evaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, pred_coco, num_classes, dataset_dir,testdev=False
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
        self.pred_coco = pred_coco
        self.num_classes = num_classes
        self.id = -1
        self.box_id = 0
        self.id_ori = 0
        self.box_id_ori = 0
        self.vid_to_coco = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "airplane"}, {"supercategorie": "", "id": 1, "name": "antelope"}, {"supercategorie": "", "id": 2, "name": "bear"}, {"supercategorie": "", "id": 3, "name": "bicycle"}, {"supercategorie": "", "id": 4, "name": "bird"}, {"supercategorie": "", "id": 5, "name": "bus"}, {"supercategorie": "", "id": 6, "name": "car"}, {"supercategorie": "", "id": 7, "name": "cattle"}, {"supercategorie": "", "id": 8, "name": "dog"}, {"supercategorie": "", "id": 9, "name": "domestic_cat"}, {"supercategorie": "", "id": 10, "name": "elephant"}, {"supercategorie": "", "id": 11, "name": "fox"}, {"supercategorie": "", "id": 12, "name": "giant_panda"}, {"supercategorie": "", "id": 13, "name": "hamster"}, {"supercategorie": "", "id": 14, "name": "horse"}, {"supercategorie": "", "id": 15, "name": "lion"}, {"supercategorie": "", "id": 16, "name": "lizard"}, {"supercategorie": "", "id": 17, "name": "monkey"}, {"supercategorie": "", "id": 18, "name": "motorcycle"}, {"supercategorie": "", "id": 19, "name": "rabbit"}, {"supercategorie": "", "id": 20, "name": "red_panda"}, {"supercategorie": "", "id": 21, "name": "sheep"}, {"supercategorie": "", "id": 22, "name": "snake"}, {"supercategorie": "", "id": 23, "name": "squirrel"}, {"supercategorie": "", "id": 24, "name": "tiger"}, {"supercategorie": "", "id": 25, "name": "train"}, {"supercategorie": "", "id": 26, "name": "turtle"}, {"supercategorie": "", "id": 27, "name": "watercraft"}, {"supercategorie": "", "id": 28, "name": "whale"}, {"supercategorie": "", "id": 29, "name": "zebra"}],
            'images': [],
            'licenses': []
        }
        self.vid_to_coco_ori = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "airplane"}, {"supercategorie": "", "id": 1, "name": "antelope"}, {"supercategorie": "", "id": 2, "name": "bear"}, {"supercategorie": "", "id": 3, "name": "bicycle"}, {"supercategorie": "", "id": 4, "name": "bird"}, {"supercategorie": "", "id": 5, "name": "bus"}, {"supercategorie": "", "id": 6, "name": "car"}, {"supercategorie": "", "id": 7, "name": "cattle"}, {"supercategorie": "", "id": 8, "name": "dog"}, {"supercategorie": "", "id": 9, "name": "domestic_cat"}, {"supercategorie": "", "id": 10, "name": "elephant"}, {"supercategorie": "", "id": 11, "name": "fox"}, {"supercategorie": "", "id": 12, "name": "giant_panda"}, {"supercategorie": "", "id": 13, "name": "hamster"}, {"supercategorie": "", "id": 14, "name": "horse"}, {"supercategorie": "", "id": 15, "name": "lion"}, {"supercategorie": "", "id": 16, "name": "lizard"}, {"supercategorie": "", "id": 17, "name": "monkey"}, {"supercategorie": "", "id": 18, "name": "motorcycle"}, {"supercategorie": "", "id": 19, "name": "rabbit"}, {"supercategorie": "", "id": 20, "name": "red_panda"}, {"supercategorie": "", "id": 21, "name": "sheep"}, {"supercategorie": "", "id": 22, "name": "snake"}, {"supercategorie": "", "id": 23, "name": "squirrel"}, {"supercategorie": "", "id": 24, "name": "tiger"}, {"supercategorie": "", "id": 25, "name": "train"}, {"supercategorie": "", "id": 26, "name": "turtle"}, {"supercategorie": "", "id": 27, "name": "watercraft"}, {"supercategorie": "", "id": 28, "name": "whale"}, {"supercategorie": "", "id": 29, "name": "zebra"}],
            'images': [],
            'licenses': []
        }
        self.testdev = testdev
        self.path = 'ILSVRC2015/Annotations/VID/'
        self.tmp_name_ori = None
        self.tmp_refined_path = pred_coco.replace('.pkl','_imdb2coco.json')
        self.gt_ori = None
        self.gt_refined = './excluded/post/vid_gt_coco.json'
        self.dataset_dir = dataset_dir

    def get_annotation(self,path):
        path = path.replace("Data","Annotations").replace("JPEG","xml")
        path = os.path.join(self.dataset_dir, path)
        if os.path.isdir(path):
            files = get_xml_list(path)
        else:
            files = [path]
        files.sort()
        anno_res = []
        for xmls in files:
            #photoname = xmls.replace("Annotations","Data").replace("xml","JPEG")
            file = minidom.parse(xmls)
            root = file.documentElement
            objs = root.getElementsByTagName("object")
            width = int(root.getElementsByTagName('width')[0].firstChild.data)
            height = int(root.getElementsByTagName('height')[0].firstChild.data)
            tempnode = []
            for obj in objs:
                nameNode = obj.getElementsByTagName("name")[0].firstChild.data
                xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
                xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
                ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
                ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
                x1 = np.max((0,xmin))
                y1 = np.max((0,ymin))
                x2 = np.min((width,xmax))
                y2 = np.min((height,ymax))
                if x2 >= x1 and y2 >= y1:
                    #tempnode.append((name_num[nameNode],x1,y1,x2,y2,))
                    tempnode.append(( x1, y1, x2, y2,name_num[nameNode],))
            num_objs = len(tempnode)
            res = np.zeros((num_objs, 5))
            for ix, obj in enumerate(tempnode):
                res[ix, 0:5] = obj[0:5]
            anno_res.append(res)
        return anno_res

    def get_std_coco(self,path,save_pth):
        # 将之前非完全COCO格式的 VID 标注转为 完全格式
        ids = []
        data_list = []
        labels_list = []
        ori_data_list = []
        ori_label_list = []

        progress_bar = tqdm if is_main_process() else iter
        vid_num = 0
        json_file = open(path)
        pre_anno = json.load(json_file)
        images = []
        for cur_img,img in enumerate(progress_bar(pre_anno['images'])):
            images.append(img)
        pre_anno['images'] = images
        json.dump(pre_anno, open(save_pth, "w"))
        pass


    def evaluate(self,):
        ids = []
        data_list = []
        labels_list = []
        ori_data_list = []
        ori_label_list = []
        score_list = []
        progress_bar = tqdm if is_main_process() else iter
        vid_num = 0
        for vid, video_preds in progress_bar(get_video_frame_iterator(self.pred_coco)): #video level
            video_preds = dict(sorted(video_preds.items(), key=lambda x: x[0]))
            for frame in video_preds.keys():  #frame level
                for ele in video_preds[frame]:#box level
                    name = ele['image_id']
                    bbox = np.array(ele['bbox'])
                    scores = ele['scores']
                    label = scores[-1]
                    cls_scores = scores[0]*scores[1]
                    pred_data = {
                        "image_id": int(self.id),
                        "category_id": int(label),
                        "bbox": bbox.tolist(),
                        "score": cls_scores.item(),
                        "segmentation": [],
                    }  # COCO json format
                    data_list.append(pred_data)
                    score_list.append(cls_scores.item())

                if not os.path.exists(self.gt_refined):
                    anno_path = os.path.join(self.path + vid , frame + '.xml')
                    anno = self.get_annotation(anno_path)
                    anno_tensor = torch.tensor(anno[0])
                    bboxes_label = xyxy2xywh(anno_tensor[:,:4])
                    cls_label = anno_tensor[:, -1]
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
                        labels_list.append(label_pred_data)
                    self.vid_to_coco['images'].append({'id': self.id})
                self.id = self.id + 1

            vid_num +=1

        plt.hist(np.array(score_list),10,range=(0,1))
        plt.show()
        if not os.path.exists(self.gt_refined):
            self.vid_to_coco['annotations'].extend(labels_list)
        else:
            self.vid_to_coco = load_json(self.gt_refined)
        eval_results = self.evaluate_prediction(data_list)
        logger.info(eval_results[-2])
        #print(eval_results)

    def evaluate_json(self,):
        ids = []
        data_list = []
        labels_list = []
        ori_data_list = []
        ori_label_list = []

        progress_bar = tqdm if is_main_process() else iter
        vid_num = 0
        json_file = open(self.pred_coco.replace('.pkl','_repp_coco.json'))
        #imdb_file = open(self.pred_coco.replace('.pkl', '_repp_imdb.txt'))
        with open(self.pred_coco.replace('.pkl', '_repp_imdb.txt'), 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        json_res = json.load(json_file)
        json_res = sorted(json_res,key=lambda k:k['image_id'])
        pre_name = json_res[0]['image_id']
        for cur_iter,video_preds in enumerate(progress_bar(json_res)): #obj level
            ele = video_preds
            name = ele['image_id']

            if (name != pre_name or cur_iter ==0):
                self.id = self.id + 1
                if not os.path.exists(self.gt_refined):
                    anno_path = self.path + name + '.xml'
                    anno = self.get_annotation(anno_path)
                    anno_tensor = torch.tensor(anno[0])
                    bboxes_label = xyxy2xywh(anno_tensor[:, :4])
                    cls_label = anno_tensor[:, -1]
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
                        labels_list.append(label_pred_data)
                    self.vid_to_coco['images'].append({'id': self.id})



            bbox = np.array(ele['bbox'])
            scores = ele['score']
            label = int(ele['category_id'])
            pred_data = {
                "image_id": int(self.id),
                "category_id": int(label),
                "bbox": bbox.tolist(),
                "score": scores,
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
            pre_name = name

        if not os.path.exists(self.gt_refined):
            self.vid_to_coco['annotations'].extend(labels_list)
        else:
            self.vid_to_coco = load_json(self.gt_refined)
        eval_results = self.evaluate_prediction(data_list)
        logger.info(eval_results[-2])

    def evaluate_prediction(self, data_dict, ori=False):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        info = "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:

            _, tmp = tempfile.mkstemp()
            if ori:
                json.dump(self.vid_to_coco_ori, open(self.gt_ori, 'w'))
                json.dump(data_dict, open(self.tmp_name_ori, 'w'))
                json.dump(self.vid_to_coco_ori, open(tmp, "w"))
            else:
                if not os.path.exists(self.gt_refined):
                    json.dump(self.vid_to_coco, open(self.gt_refined, 'w'))
                json.dump(data_dict, open(self.tmp_refined_path, 'w'))
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
            map_info_list = []
            for i in range(len(vid_classes)):
                stats, _ = cocoEval.m_summarize(catId=i)
                map_info_list.append("{:15}:{}".format(vid_classes[i], stats[1]))
            res = '\n'.join(map_info_list)
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]

            AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            info += "per class AP:\n" + AP_table + "\n"
            info += "per class AR:\n" + AR_table + "\n"

            return cocoEval.stats[0], cocoEval.stats[1], info,res
        else:
            return 0, 0, info

    def ana_distribution(self,):
        ids = []
        data_list = []
        labels_list = []
        score_list = []
        fg_list = []
        cls_list = []
        progress_bar = tqdm if is_main_process() else iter
        vid_num = 0
        for vid, video_preds in get_video_frame_iterator(self.pred_coco): #video level
            video_preds = dict(sorted(video_preds.items(), key=lambda x: x[0]))
            for frame in video_preds.keys():  #frame level
                for ele in video_preds[frame]:#box level
                    scores = ele['scores']
                    cls_scores = scores[0]*scores[1]
                    fg_list.append(scores[0])
                    cls_list.append(scores[1])
                    score_list.append(cls_scores.item())
            print(str(vid_num) + '/555')
            vid_num +=1

        plt.hist(np.array(score_list),10,range=(0,1))
        plt.title('Histogram of score')
        plt.show()
        plt.savefig('./score_dist.jpg')

        plt.hist(np.array(fg_list),10,range=(0,1))
        plt.title('Histogram of foreground score')
        plt.show()
        plt.savefig('./score_fg_dist.jpg')
        plt.hist(np.array(cls_list),10,range=(0,1))
        plt.title('Histogram of classs score')
        plt.show()
        plt.savefig('./score_cls_dist.jpg')


if __name__ == "__main__":
    args = make_parser().parse_args()
    print('processing {}'.format(args.path))
    debug = REPP_to_COCO_Evaluator(args.path,len(vid_classes),args.dataset_dir)
    debug.evaluate()
    if args.tide:
        from tidecv import TIDE, datasets
        print('loading coco format GT and Det...')
        tide = TIDE()
        bbox_results = datasets.COCOResult(args.path.replace('.pkl','_imdb2coco.json'))  # These files were downloaded above.
        gt = datasets.COCO('./excluded/post/vid_gt_coco.json')
        print('Evaluating different range errors...')
        tide.evaluate_range(gt, bbox_results, mode=TIDE.BOX)  # Use TIDE.MASK for masks
        tide.summarize()  # Summarize the results as tables in the console
        tide.plot(os.path.dirname(args.path))


    # debug.ana_distribution()
    #debug.get_std_coco('../annotations/annotations_val_coco.json',save_pth='../annotations/annotations_val_coco.json')
    #COCO_eva = REPP_to_COCO_Evaluator(args.path,30,args)
    #debug.evaluate()
