#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os
import random

import numpy
from loguru import logger

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.sampler import Sampler,BatchSampler,SequentialSampler
from xml.dom import minidom
import math
from yolox.utils import xyxy2cxcywh

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".JPEG"]
XML_EXT = [".xml"]
name_list = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']
numlist = range(30)
name_num = dict(zip(name_list,numlist))

class VIDDataset(torchDataset):
    """
    VID sequence
    """

    def __init__(
        self,
        file_path="train_seq.npy",
        img_size=(416, 416),
        preproc=None,
        lframe = 18,
        gframe = 6,
        val = False,
        mode='random',
        dataset_pth = '',
        tnum = 1000
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__()
        self.tnum = tnum
        self.input_dim = img_size
        self.file_path = file_path
        self.mode = mode  # random, continous, uniform
        self.img_size = img_size
        self.preproc = preproc
        self.val = val
        self.res = self.photo_to_sequence(self.file_path,lframe,gframe)
        self.dataset_pth = dataset_pth

    def __len__(self):
        return len(self.res)


    def photo_to_sequence(self,dataset_path,lframe,gframe):
        '''

        Args:
            dataset_path: list,every element is a list contain all frames in a video dir
        Returns:
            split result
        '''
        res = []
        dataset = np.load(dataset_path,allow_pickle=True).tolist()
        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    split_num = int(ele_len / (gframe))
                    random.shuffle(element)
                    for i in range(split_num):
                        res.append(element[i * gframe:(i + 1) * gframe])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)
        # test = []
        # for ele in res:
        #     test.extend(ele)
        # random.shuffle(test)
        # i = 0
        # for ele in res:
        #     for j in range(gframe):
        #         ele[j] = test[i]
        #         i += 1

        if self.val:
            random.seed(42)
            random.shuffle(res)
            if self.tnum == -1:
                return res
            else:
                return res[:self.tnum]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res[:15000]


    def get_annotation(self,path,test_size):
        path = path.replace("Data","Annotations").replace("JPEG","xml")
        if os.path.isdir(path):
            files = get_xml_list(path)
        else:
            files = [path]
        files.sort()
        anno_res = []
        for xmls in files:
            photoname = xmls.replace("Annotations","Data").replace("xml","JPEG")
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
            r = min(test_size[0] / height, test_size[1] / width)
            for ix, obj in enumerate(tempnode):
                res[ix, 0:5] = obj[0:5]
            res[:, :-1] *= r
            anno_res.append(res)
        return anno_res


    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        path = os.path.join(self.dataset_pth,path)
        annos = self.get_annotation(path, self.img_size)[0]

        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos, img_info, path

    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path



def get_xml_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in XML_EXT:
                image_names.append(apath)

    return image_names

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def make_path(train_dir,save_path):
    res = []
    for root,dirs,files in os.walk(train_dir):
        temp = []
        for filename in files:
            apath = os.path.join(root, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                temp.append(apath)
        if(len(temp)):
            temp.sort()
            res.append(temp)
    res_np = np.array(res,dtype=object)
    np.save(save_path,res_np)


class TestSampler(SequentialSampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class TrainSampler(Sampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.data_source.res)
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class VIDBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            for filename in ele:
                batch.append(filename)
                if (len(batch)) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch)>0 and not self.drop_last:
            yield batch
    def __len__(self):
        return len(self.sampler)

class VIDBatchSampler_Test(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            yield ele
            # for filename in ele:
            #     batch.append(filename)
            #     if (len(batch)) == self.batch_size:
            #         yield batch
            #         batch = []
            # if len(batch)>0 and not self.drop_last:
            #     yield batch
    def __len__(self):
        return len(self.sampler)

def collate_fn(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([120,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        #path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    # path_sequence= torch.tensor(path_sequence)
    # time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,None

def get_vid_loader(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader

def vid_val_loader(batch_size,data_num_workers,dataset,):
    sampler = VIDBatchSampler_Test(TestSampler(dataset),batch_size,drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': collate_fn
    }
    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return loader

def collate_fn_trans(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([100,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(copy.deepcopy(sample[1])))
        sample[1][:,1:]=xyxy2cxcywh(sample[1][:,1:])
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    path_sequence= torch.tensor(path_sequence)
    time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,time_embedding

def get_trans_loader(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.max_iter = len(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target,_,_,_,self.time_ebdding = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.time_ebdding = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        time_ebdding = self.time_ebdding
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target,time_ebdding

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

def get_timing_signal_1d(index_squence,channels,min_timescale=1.0, max_timescale=1.0e4,):
    num_timescales = channels // 2

    log_time_incre = torch.tensor(math.log(max_timescale/min_timescale)/(num_timescales-1))
    inv_timescale = min_timescale*torch.exp(torch.arange(0,num_timescales)*-log_time_incre)

    scaled_time = torch.unsqueeze(index_squence,1)*torch.unsqueeze(inv_timescale,0) #(index_len,1)*(1,channel_num)
    sig = torch.cat([torch.sin(scaled_time),torch.cos(scaled_time)],dim=1)
    return sig
