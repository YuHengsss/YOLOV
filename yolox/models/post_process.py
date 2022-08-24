import copy
import torch
import torchvision
import random

def postprocess(prediction, num_classes, fc_outputs, conf_thre=0.001, nms_thre=0.5):
    output = [None for _ in range(len(prediction))]
    output_ori = [None for _ in range(len(prediction))]
    prediction_ori = copy.deepcopy(prediction)
    cls_conf, cls_pred = torch.max(fc_outputs, -1, keepdim=False)

    for i, detections in enumerate(prediction):

        if not detections.size(0):
            continue

        detections[:, 5] = cls_conf[i].sigmoid()
        detections[:, 6] = cls_pred[i]
        tmp_cls_score = fc_outputs[i].sigmoid()
        cls_mask = tmp_cls_score >= conf_thre
        cls_loc = torch.where(cls_mask)
        scores = torch.gather(tmp_cls_score[cls_loc[0]],dim=-1,index=cls_loc[1].unsqueeze(1))#[:,cls_loc[1]]#tmp_cls_score[torch.stack(cls_loc).T]#torch.gather(tmp_cls_score, dim=1, index=torch.stack(cls_loc).T)

        detections[:, -num_classes:] = tmp_cls_score
        detections_raw = detections[:, :7]
        new_detetions = detections_raw[cls_loc[0]]
        new_detetions[:, -1] = cls_loc[1]
        new_detetions[:,5] = scores.squeeze()
        detections_high = new_detetions  # new_detetions
        detections_ori = prediction_ori[i]
        #print(len(detections_high.shape))

        conf_mask = (detections_high[:, 4] * detections_high[:, 5] >= conf_thre).squeeze()
        detections_high = detections_high[conf_mask]

        if not detections_high.shape[0]:
            continue
        if len(detections_high.shape)==3:
            detections_high = detections_high[0]
        nms_out_index = torchvision.ops.batched_nms(
            detections_high[:, :4],
            detections_high[:, 4] * detections_high[:, 5],
            detections_high[:, 6],
            nms_thre,
        )

        detections_high = detections_high[nms_out_index]
        output[i] = detections_high
        detections_ori = detections_ori[:, :7]
        conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
        detections_ori = detections_ori[conf_mask]
        nms_out_index = torchvision.ops.batched_nms(
            detections_ori[:, :4],
            detections_ori[:, 4] * detections_ori[:, 5],
            detections_ori[:, 6],
            nms_thre,
        )

        detections_ori = detections_ori[nms_out_index]
        output_ori[i] = detections_ori

    return output, output_ori


def postprocess_pure(prediction, num_classes, fc_outputs, conf_thre=0.001, nms_thre=0.5):
    output = [None for _ in range(len(prediction))]
    output_ori = [None for _ in range(len(prediction))]
    prediction_ori = copy.deepcopy(prediction)
    cls_conf, cls_pred = torch.max(fc_outputs, -1, keepdim=False)

    for i, detections in enumerate(prediction):

        if not detections.size(0):
            continue

        detections[:, 5] = cls_conf[i].sigmoid()
        detections[:, 6] = cls_pred[i]
        tmp_cls_score = fc_outputs[i].sigmoid()
        cls_mask = tmp_cls_score >= conf_thre
        cls_loc = torch.where(cls_mask)
        scores = torch.gather(tmp_cls_score[cls_loc[0]],dim=-1,index=cls_loc[1].unsqueeze(1))#[:,cls_loc[1]]#tmp_cls_score[torch.stack(cls_loc).T]#torch.gather(tmp_cls_score, dim=1, index=torch.stack(cls_loc).T)

        detections[:, -num_classes:] = tmp_cls_score
        detections_raw = detections[:, :7]
        new_detetions = detections_raw[cls_loc[0]]
        new_detetions[:, -1] = cls_loc[1]
        new_detetions[:,5] = scores.squeeze()
        detections_high = new_detetions  # new_detetions
        detections_ori = prediction_ori[i]
        #print(len(detections_high.shape))

        conf_mask = (detections_high[:, 4] * detections_high[:, 5] >= conf_thre).squeeze()
        detections_high = detections_high[conf_mask]

        if not detections_high.shape[0]:
            continue
        if len(detections_high.shape)==3:
            detections_high = detections_high[0]
        nms_out_index = torchvision.ops.batched_nms(
            detections_high[:, :4],
            detections_high[:, 4] * detections_high[:, 5],
            detections_high[:, 6],
            nms_thre,
        )

        detections_high = detections_high[nms_out_index]
        output[i] = detections_high


    return output


def visual_sim(attn,imgs,simN,predictions,cos_sim):
    sort_res = torch.sort(attn,descending=True)
    #cos_sim_res = torch.sort(cos_sim,descending=True)
    img = imgs[0,:,:,:].permute(1,2,0)
    support_idx = sort_res.indices[:simN, :]
    for bidx in range(int(5)):
        box = predictions[0][bidx,:4]
        visual_pred(img,box,title='key proposal'+str(bidx))
        for sidx in range(6):
            frame_idx = support_idx[bidx,sidx]
            fth = int(frame_idx/simN)
            bth = int(frame_idx%simN)
            simg = imgs[fth,:,:,:].permute(1,2,0)
            box = predictions[fth][bth,:4]
            visual_pred(simg,box,'key-'+str(bidx)+'-support'\
                        +str(sidx)+"-"+str(float(sort_res.values[bidx,sidx]))[:4]\
                        + '-'+str(float(cos_sim[bidx,frame_idx]))[:4])
        if bidx>=5:
            break

def visual_pred(img,box,title=''):
    import cv2
    import numpy
    import matplotlib.pyplot as plt
    import time

    img = img.cpu()
    img = img.detach().numpy()
    #img = cv2.resize(img,[360,640])
    x0 = max(int(box[0]),0)
    y0 = max(int(box[1]),0)
    x1 = int(box[2])
    y1 = int(box[3])
    img = numpy.array(img, dtype=numpy.int)
    res = img[y0:y1,x0:x1,:]
    b,g,r = cv2.split(res)
    res = cv2.merge([r,g,b])

    fig = plt.figure(title)
    plt.imshow(res)
    plt.title(title)

    plt.savefig('/home/tuf/yh/YOLOV/visual_fandc_no2/'+str(time.time())+'.png')
    #plt.show()
    plt.close(fig)
    return

def online_previous_selection(tmp_bank, frame_num=31,local = True,local_bank = []):
    '''

    :param tmp_bank: list [[idx] [result] [cls] [reg] [linear0]]
    :return:dict {'pred_idx':[],'cls_feature':Tensor,'reg_feature':Tensor,'pred_result':list,'linear0_feature'}
    '''
    res = {}
    if len(tmp_bank[0]) < 2: return []
    # tmp = list(zip(tmp_bank[0], tmp_bank[1], tmp_bank[2], tmp_bank[3], tmp_bank[4]))
    # shuffle(tmp)
    for i in range(len(tmp_bank)):
        random.seed(42)
        random.shuffle(tmp_bank[i])

    res['cls_feature'] = torch.cat(tmp_bank[0][:frame_num], dim=0)
    res['reg_feature'] = torch.cat(tmp_bank[1][:frame_num], dim=0)
    res['cls_scores'] = torch.cat(tmp_bank[2][:frame_num])
    res['reg_scores'] = torch.cat(tmp_bank[3][:frame_num])
    res['local_results'] = []
    if local:
        if local_bank[0] != []:
            msa = torch.cat(local_bank[0][:frame_num], dim=0)
            res['local_results'] = {}
            local_num =len(local_bank[0][-20:])
            res['local_results']['cls_scores'] = torch.cat(local_bank[2][-local_num:])
            res['local_results']['reg_scores'] =  torch.cat(local_bank[3][-local_num:])
            res['local_results']['msa'] =  torch.cat(local_bank[0][-local_num:], dim=0)
            res['local_results']['boxes'] =  torch.cat(local_bank[1][-local_num:], dim=0)
        else:
            res['local_results'] = []
    return res