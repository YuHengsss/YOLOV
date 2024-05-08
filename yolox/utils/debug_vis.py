import copy
import cv2
from .visualize import vis
from .box_op import box_cxcywh_to_xyxy
from yolox.data.datasets.vid_classes import VID_classes
import torch
from matplotlib import pyplot as plt
import numpy as np
def visual_predictions(img,predictions,xyxy=True,batch_idx=0,class_num=30,classes=VID_classes,conf_thres=0.1):
    #convert img from bchw to bhwc for vis
    img = img.permute(0,2,3,1).cpu().numpy()
    img = img[batch_idx]
    #convert float32 to the format which cv2 can handle
    img = img.astype('uint8')
    #clip img to [0,255]
    img = np.clip(img,0,255)
    boxes = predictions[batch_idx]
    pred_conf = predictions[batch_idx][:,4]
    pred_cls_score = predictions[batch_idx][:,5]
    pred_score = pred_conf * pred_cls_score
    pred_cls_id = torch.argmax(predictions[batch_idx][:,-class_num:],dim=1)
    mask = pred_score > conf_thres
    if not xyxy:
        boxes = box_cxcywh_to_xyxy(boxes)
    #clip boxes the img size
    boxes[:,0] = torch.clamp(boxes[:,0],0,img.shape[1])
    boxes[:,1] = torch.clamp(boxes[:,1],0,img.shape[0])
    boxes[:,2] = torch.clamp(boxes[:,2],0,img.shape[1])
    boxes[:,3] = torch.clamp(boxes[:,3],0,img.shape[0])
    img = np.ascontiguousarray(img)
    img_vis = vis(copy.deepcopy(img),boxes[mask],pred_score[mask],pred_cls_id[mask],conf=conf_thres,class_names=classes,t_size=0.4)
    #convert bgr to rgb
    img_vis = img_vis[:,:,::-1]
    plt.figure(figsize=(10,10))
    plt.imshow(img_vis)
    plt.show()
    return img_vis