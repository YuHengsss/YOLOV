#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:46:22 2020

@author: asabater
"""

import pickle

import numpy
import numpy as np
from scipy import signal, ndimage
import json

INF = 9e15

from repp_utils import  get_iou, get_pair_features
# =============================================================================
# Robust and Efficient Post-Processing for Video Object Detection (REPP)
# =============================================================================


class REPP():

    def __init__(self, min_tubelet_score, add_unmatched, min_pred_score,
                 distance_func, clf_thr, clf_mode, appearance_matching,
                 recoordinate, recoordinate_std,weight_path
                 ):

        self.min_tubelet_score = min_tubelet_score  # threshold to filter out low-scoring tubelets
        self.min_pred_score = min_pred_score  # threshold to filter out low-scoring base predictions
        self.add_unmatched = add_unmatched  # True to add unlinked detections to the final set of detections. Leads to a lower mAP

        self.distance_func = distance_func  # LogReg to use the learning-based linking model. 'def' to use the baseline from SBM
        self.clf_thr = clf_thr  # threshold to filter out detection linkings
        self.clf_mode = clf_mode  # Relation between the logreg score and the semmantic similarity. 'dot' recommended
        self.appearance_matching = appearance_matching  # True to use appearance similarity features

        self.recoordinate = recoordinate  # True to perform a recordinating step
        self.recoordinate_std = recoordinate_std  # Strength of the recoordinating step
        self.store_coco = True
        #self.store_imdb = True#store_imdb  # True to store predictions with the IMDB format. Needed for evaluation

        if self.distance_func == 'def':
            self.match_func = self.distance_def
        elif self.distance_func == 'logreg':
            self.clf_match, self.matching_feats = pickle.load(
                open(weight_path, 'rb'))
            self.match_func = self.distance_logreg
        else:
            raise ValueError('distance_func not recognized:', self.distance_func)

    def distance_def(self, p1, p2):
        iou = get_iou(p1['bbox'][:], p2['bbox'][:])
        score = np.dot(p1['scores'], p2['scores'])
        div = iou * score
        if div == 0: return INF
        return 1 / div

    # Computes de linking score between a pair of detections
    def distance_logreg(self, p1, p2):
        pair_features = get_pair_features(p1, p2, self.matching_feats)  # , image_size[0], image_size[1]
        score = self.clf_match.predict_proba(np.array([[pair_features[col] for col in self.matching_feats]]))[:, 1]
        if score < self.clf_thr: return INF

        if self.clf_mode == 'max':
            score = p1['scores'].max() * p2['scores'].max() * score
        elif self.clf_mode == 'dot':
            score = np.dot(p1['scores'], p2['scores']) * score
        elif self.clf_mode == 'dot_plus':
            score = np.dot(p1['scores'], p2['scores']) + score
        elif self.clf_mode == 'def':
            return distance_def(p1, p2)
        elif self.clf_mode == 'raw':
            pass
        else:
            raise ValueError('error post_clf')
        return 1 - score

    # Return a list of pairs of frames lnked accross frames
    def get_video_pairs(self, preds_frame):
        num_frames = len(preds_frame)
        frames = list(preds_frame.keys())
        frames = sorted(frames, key=int)

        pairs, unmatched_pairs = [], []
        for i in range(num_frames - 1):

            pairs_i = []
            frame_1, frame_2 = frames[i], frames[i + 1]
            preds_frame_1, preds_frame_2 = preds_frame[frame_1], preds_frame[frame_2]
            num_preds_1, num_preds_2 = len(preds_frame_1), len(preds_frame_2)

            # Any frame has no preds -> save empty pairs

            if num_preds_1 != 0 and num_preds_2 != 0:
                # Get distance matrix
                distances = np.zeros((num_preds_1, num_preds_2))
                for i, p1 in enumerate(preds_frame_1):
                    for j, p2 in enumerate(preds_frame_2):
                        distances[i, j] = self.match_func(p1, p2)

                # Get frame pairs
                pairs_i = self.solve_distances_def(distances, maximization_problem=False)

            unmatched_pairs_i = [i for i in range(num_preds_1) if i not in [p[0] for p in pairs_i]]
            pairs.append(pairs_i);
            unmatched_pairs.append(unmatched_pairs_i)

        return pairs, unmatched_pairs

    # Solve distance matrix and return a list of pair of linked detections from two consecutive frames
    def solve_distances_def(self, distances, maximization_problem):
        pairs = []
        if maximization_problem:
            while distances.min() != -1:
                inds = np.where(distances == distances.max())
                a, b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
                a, b = int(a), int(b)
                pairs.append((a, b))
                distances[a, :] = -1
                distances[:, b] = -1
        else:
            while distances.min() != INF:
                inds = np.where(distances == distances.min())
                a, b = inds if len(inds[0]) == 1 else (inds[0][0], inds[1][0])
                a, b = int(a), int(b)
                pairs.append((a, b))
                distances[a, :] = INF
                distances[:, b] = INF

        return pairs

        # Create tubelets from list of linked pairs

    def get_tubelets(self, preds_frame, pairs):

        num_frames = len(preds_frame)
        frames = list(preds_frame.keys())
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
                        tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
                        tubelets_count += 1
                        ind = None
                        break

                        # Continue tubelet
                    else:
                        pair = pair[0];
                        del pairs[current_frame][pairs[current_frame].index(pair)]
                        tubelets[tubelets_count].append((current_frame, preds_frame[frames[current_frame]][ind]))
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
                        tubelets.append([(current_frame,
                                          preds_frame[frames[current_frame]][pair[0]])])
                        ind = pair[1]

            # Tubelet has finished in the last frame
            if ind != None:
                tubelets[tubelets_count].append((current_frame + 1, preds_frame[frames[current_frame + 1]][ind]))  # 4
                tubelets_count += 1
                ind = None

        return tubelets

    # Performs the re-scoring refinment
    def rescore_tubelets(self, tubelets):
        for t_num in range(len(tubelets)):
            t_scores = [p['scores'] for _, p in tubelets[t_num]]
            new_scores = np.mean(t_scores, axis=0)
            for i in range(len(tubelets[t_num])): tubelets[t_num][i][1][
                'scores'] = new_scores  # np.mean(t_scores[:i+1], axis=0)
            for i in range(len(tubelets[t_num])):
                if 'emb' in tubelets[t_num][i][1]: del tubelets[t_num][i][1]['emb']

        return tubelets

    # Performs de re-coordinating refinment
    def recoordinate_tubelets_full(self, tubelets, ms=-1):
        if ms == -1: ms = 40
        for t_num in range(len(tubelets)):
            t_coords = np.array([p['bbox'] for _, p in tubelets[t_num]])
            w = signal.gaussian(len(t_coords) * 2 - 1, std=self.recoordinate_std * 100 / ms)
            w /= sum(w)

            for num_coord in range(4):
                t_coords[:, num_coord] = ndimage.convolve(t_coords[:, num_coord], w, mode='reflect')

            for num_bbox in range(len(tubelets[t_num])):
                tubelets[t_num][num_bbox][1]['bbox'] = t_coords[num_bbox, :].tolist()

        return tubelets

    # Extracts predictions from tubelets
    def tubelets_to_predictions(self, tubelets_video, preds_format):

        preds, track_id_num = [], 0
        for tub in tubelets_video:
            for _, pred in tub:
                for cat_id, s in enumerate(pred['scores']):
                    if s < self.min_pred_score: continue
                    # if s != max(pred['scores']): continue
                    if preds_format == 'coco':
                        preds.append({
                            'image_id': pred['image_id'],
                            'bbox': list(map(float, pred['bbox'])),
                            'score': float(s),
                            'category_id': cat_id,
                            'track_id': track_id_num,
                        })
                    else:
                        raise ValueError('Predictions format not recognized')

            track_id_num += 1
        return preds

    def add_unmatched_pairs_as_single_tubelet(self, unmatched_pairs, video_predictions):
        res_list = []
        list_pred = list(video_predictions.values())
        for i in range(len(unmatched_pairs)):
            if unmatched_pairs[i] != []:
                for ele in unmatched_pairs[i]:
                    res_list.append([(i, list_pred[i][ele])])
        return res_list

    def __call__(self, video_predictions):
        # Filter out low-score predictions
        for frame in video_predictions.keys():
            tmp = []
            for p in video_predictions[frame]:
                idx, scores = int(p['scores'][2]), p['scores'][0] * p['scores'][1]
                if scores >= self.min_tubelet_score:
                    p['scores'] = numpy.zeros([30])
                    p['scores'][idx] = scores
                    tmp.append(p)

            video_predictions[frame] = tmp#[p for p in video_predictions[frame] ]


        pairs, unmatched_pairs = self.get_video_pairs(video_predictions)
        tubelets = self.get_tubelets(video_predictions, pairs)

        tubelets = self.rescore_tubelets(tubelets)
        if self.recoordinate: tubelets = self.recoordinate_tubelets_full(tubelets)

        if self.add_unmatched:
            # print('Adding unmatched')
            tubelets += self.add_unmatched_pairs_as_single_tubelet(unmatched_pairs, video_predictions)

        predictions_coco = self.tubelets_to_predictions(tubelets, 'coco')

        return predictions_coco



