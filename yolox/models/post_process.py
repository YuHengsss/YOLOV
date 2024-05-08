import copy
import torch
import torchvision
import random
import time
from yolox.utils import bboxes_iou
def postprocess(prediction, num_classes, fc_outputs,
                conf_output, conf_thre=0.001, nms_thre=0.5,
                cls_sig=True,return_idx=False):
    output = [None for _ in range(len(prediction))]
    output_ori = [None for _ in range(len(prediction))]
    prediction_ori = copy.deepcopy(prediction)
    cls_pred, cls_conf = [],[]
    for _ in range(len(prediction)):
        tmp_cls,tmp_pred = torch.max(fc_outputs[_], -1, keepdim=False) #
        cls_pred.append(tmp_pred)
        cls_conf.append(tmp_cls)
    # cls_conf, cls_pred = torch.max(fc_outputs, -1, keepdim=False) #
    nms_out_idxs = []
    for i, detections in enumerate(prediction):

        if detections==None or not detections.size(0):
            continue
        if conf_output is not None:
            detections[:, 4] = conf_output[i].sigmoid()
        detections[:, 5] = cls_conf[i].sigmoid()
        detections[:, 6] = cls_pred[i]
        if cls_sig:
            tmp_cls_score = fc_outputs[i].sigmoid()
        else:
            tmp_cls_score = fc_outputs[i]
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
        conf_mask = detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre

        #conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
        detections_ori = detections_ori[conf_mask]

        nms_out_index = torchvision.ops.batched_nms(
            detections_ori[:, :4],
            detections_ori[:, 4] * detections_ori[:, 5],
            detections_ori[:, 6],
            nms_thre,
        )

        detections_ori = detections_ori[nms_out_index]
        output_ori[i] = detections_ori
        if return_idx:
            nms_out_idxs.append(nms_out_index + i*detections.size(0))
    if return_idx: return output, output_ori,torch.cat(nms_out_idxs,dim=0)
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

INF = 9e15

def match_func(p1, p2, iou_thr=0.5):
    iou = bboxes_iou(p1['bbox'][:], p2['bbox'][:])
    if iou < iou_thr: return INF
    score = torch.dot(p1['scores'], p2['scores'])
    div = iou * score
    if div == 0: return INF
    return 1 / div


def solve_distances_def(distances, maximization_problem=False):
    pairs = []
    while distances.max() != -1:
        inds = torch.where(distances == distances.max())
        a, b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
        a, b = int(a), int(b)
        pairs.append((a, b))
        distances[a, :] = -1
        distances[:, b] = -1
    return pairs

def get_video_pairs(distance_list):
    pairs, unmatched_pairs = [], []
    for distances in distance_list:
        pairs_i = solve_distances_def(distances, maximization_problem=True)
        pairs.append(pairs_i)
    return pairs


def get_tubelets(distance_list, pairs):

    num_frames = len(distance_list)+1#len(preds_frame)
    tubelets, tubelets_count = [], 0

    first_frame = 0

    while first_frame != num_frames - 1:
        ind = None
        for current_frame in range(first_frame, num_frames - 1):

            # Continue tubelet
            if ind is not None:
                pair = [p for p in pairs[current_frame] if p[0] == ind]
                # Tubelet ended
                if len(pair) == 0:
                    tubelets[tubelets_count].append(current_frame*30+ind)
                    # tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
                    tubelets_count += 1
                    ind = None
                    break

                    # Continue tubelet
                else:
                    pair = pair[0]
                    del pairs[current_frame][pairs[current_frame].index(pair)]
                    tubelets[tubelets_count].append(current_frame*30+ind)
                    # tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
                    ind = pair[1]

            # Looking for a new tubelet
            else:
                # No more candidates in current frame -> keep searching
                if len(pairs[current_frame]) == 0:
                    first_frame = current_frame + 1
                    continue
                # Beginning a new tubelet in current frame
                else:
                    pair = pairs[current_frame][0];
                    del pairs[current_frame][0]
                    # tubelets.append([(current_frame,
                    #                   preds_frame[frames[current_frame]][pair[0]])])
                    tubelets.append([current_frame*30+pair[0]])
                    ind = pair[1]

        # Tubelet has finished in the last frame
        if ind != None:
            tubelets[tubelets_count].append((current_frame + 1)*30+ind)
            # tubelets[tubelets_count].append((current_frame + 1, preds_frame[frames[current_frame + 1]][ind]))  # 4
            tubelets_count += 1
            ind = None

    return tubelets

def get_linking_mat(distance_list,P=30,LF=16):
    pairs = get_video_pairs(distance_list)
    tubelets = get_tubelets(distance_list, pairs)
    linking_mat = torch.zeros(P*LF,P*LF)
    for tubelet in tubelets:
        tmp_tub = torch.tensor(tubelet)#N
        for idx in tmp_tub:
            idx = torch.tensor(idx).repeat(len(tmp_tub))#N
            coords = torch.stack([idx,tmp_tub],dim=-1) #N,2
            linking_mat[coords[:,0],coords[:,1]] = 1
    # #merge the linking mat and its transpose
    # linking_mat = linking_mat + linking_mat.T
    linking_mat[torch.arange(P*LF),torch.arange(P*LF)] = 1
    return linking_mat,tubelets


def post_linking(fc_outputs,adj_lists,pred_results,P,Cls,names,exp):
    stime = time.time()
    fc_output = torch.cat(fc_outputs, dim=0)

    max_linking_frames = 400
    splits =  int(len(fc_output) / max_linking_frames) + 1
    split_frame = int(len(fc_output) / splits) + 1
    results = []
    for i in range(splits):
        fc_output_split = fc_output[i * split_frame:(i + 1) * split_frame]
        pred_results_split = pred_results[i * split_frame:(i + 1) * split_frame]
        adj_lists_split = adj_lists[i * split_frame:(i + 1) * split_frame-1]
        linking_mat, tubelets = get_linking_mat(adj_lists_split, P, len(pred_results_split))
        linking_mat = linking_mat.type_as(fc_output_split)
        fc_output_copy = fc_output_split.clone().view(-1, Cls).sigmoid()  # 480,30
        fc_output_copy = (linking_mat @ fc_output_copy) / torch.sum(linking_mat, 1, keepdim=True)
        fc_output_split = fc_output_copy.view(-1, P, Cls)
        pred_result = torch.cat(pred_results_split, dim=0)
        pred_conf = pred_result[:, 4].unsqueeze(-1).type_as(fc_output_split)  # 480*1
        pred_conf = (linking_mat @ pred_conf) / torch.sum(linking_mat, 1, keepdim=True)
        pred_result[:, 4] = pred_conf.squeeze(-1)
        pred_result = [pred_result[i * P:(i + 1) * P] for i in range(len(pred_results_split))]
        result, _ = postprocess(copy.deepcopy(pred_result), Cls, fc_output_split, None, nms_thre=exp.nmsthre, cls_sig=False)
        results.extend(result)

    # linking_mat, tubelets = get_linking_mat(adj_lists, P, len(pred_results))
    # linking_mat = linking_mat.type_as(fc_output)
    # fc_output_copy = fc_output.clone().view(-1, Cls).sigmoid()  # 480,30
    # fc_output_copy = (linking_mat @ fc_output_copy) / torch.sum(linking_mat, 1, keepdim=True)
    # fc_output = fc_output_copy.view(-1, P, Cls)
    # pred_result = torch.cat(pred_results, dim=0)
    # pred_conf = pred_result[:, 4].unsqueeze(-1).type_as(fc_output)  # 480*1
    # pred_conf = (linking_mat @ pred_conf) / torch.sum(linking_mat, 1, keepdim=True)
    # pred_result[:, 4] = pred_conf.squeeze(-1)
    # pred_result = [pred_result[i * P:(i + 1) * P] for i in range(len(pred_results))]
    # result, _ = postprocess(copy.deepcopy(pred_result), Cls, fc_output, None, nms_thre=exp.nmsthre, cls_sig=False)
    endtime = time.time()
    #print('Post process time:{},total frames:{} '.format(endtime - stime, len(names)))
    return results


def postprocess_widx(prediction, num_classes=30, conf_thre=0.01, nms_thre=0.5):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    output_index = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        #conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )

        #print(nms_out_index.shape)
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
        output_index[i] = nms_out_index

    return output, output_index

def find_idx(prediction, num_classes=30, conf_thre=0.001, nms_thre=0.5):
    box_corner = prediction.new(prediction.shape) #　prediction in cxcxywh format
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    output_index = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        #conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )

        #print(nms_out_index.shape)
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
        output_index[i] = nms_out_index

    return output, output_index

def find_features(cls_features,reg_features,idxs,):
    features_cls_rec = []
    features_reg_rec = []
    for i, feature in enumerate(cls_features):
        features_cls_rec.append(cls_features[i, idxs[i]])
        features_reg_rec.append(reg_features[i, idxs[i]])
    features_cls_rec = torch.cat(features_cls_rec)
    features_reg_rec = torch.cat(features_reg_rec)
    return features_cls_rec, features_reg_rec


def postpro_woclass(prediction, num_classes, nms_thre=0.75, topK=30, ota_idxs=None):
    # find topK predictions, play the same role as RPN
    '''

    Args:
        prediction: [batch,feature_num,5+clsnum]
        num_classes:
        conf_thre:
        conf_thre_high:
        nms_thre:

    Returns:
        [batch,topK,5+clsnum]
    '''
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    output_index = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        #take ota idxs as output in training mode
        if ota_idxs is not None and len(ota_idxs[i]) > 0:
            ota_idx = ota_idxs[i]
            topk_idx = torch.stack(ota_idx).type_as(image_pred)
            output[i] = image_pred[topk_idx, :]
            output_index[i] = topk_idx
            continue

        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)

        conf_score = image_pred[:, 4]
        top_pre = torch.topk(conf_score, k=750)
        sort_idx = top_pre.indices[:750]
        detections_temp = detections[sort_idx, :]
        nms_out_index = torchvision.ops.batched_nms(
            detections_temp[:, :4],
            detections_temp[:, 4] * detections_temp[:, 5],
            detections_temp[:, 6],
            nms_thre,
        )

        topk_idx = sort_idx[nms_out_index[:topK]]
        output[i] = detections[topk_idx, :]
        output_index[i] = topk_idx

    return output, output_index


def post_threhold(prediction, num_classes=30, conf_thre=0.001):
    box_corner = prediction.new(prediction.shape) #　prediction in cxcxywh format
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    output_index = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        #conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        output[i] = detections


    return output

