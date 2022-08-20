#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:38:36 2020

@author: asabater
"""

import json
from tqdm import tqdm
import os
import scipy.io as sio
import numpy as np



def coco_preds_to_imdb(preds_filename, imageset_filename, store_filename=None):
	"""
	Convert predictions in COCO format to IMDB format
	Each ouput line represent an image and its predictions and has the format: 
		img_id object_labels object_confs object_boxes
	
	Parameters
	----------
	preds_filename : str
		COCO predictions file path. Must be a .json file.
	imageset_filename : str
		IMDB ImageSet path.
	store_filename : str, optional
		Path where to store converted predictions. Must be a .txt file. The default is None.

	Returns
	-------
	store_filename : str
		Path where to store converted predictions. Must be a .txt file. The default is None.
	"""

	if store_filename is None:
		store_filename = preds_filename.replace('.json', '_imdb.txt')

	if os.path.isfile(store_filename): 
		print('IMDB predictions already computed:', store_filename)
		return store_filename
	print('Computing IMDB predictions')

	preds_orig = json.load(open(preds_filename, 'r'))
	
	with open(imageset_filename, 'r') as f: image_set = f.read().splitlines()
	image_set = { l.split()[0]:int(l.split()[1]) for l in image_set }
	
	imdb_lines = []
#	for p in tqdm(preds_orig):
	for p in preds_orig:
	    imdb_lines.append('{} {} {} {} {} {} {}'.format(
	            image_set['/'.join(p['image_id'].split('/')[-2:])],
	            # image_set[p['image_id']],
	            p['category_id']+1,
	            p['score'],
	            p['bbox'][0], p['bbox'][1], p['bbox'][0]+p['bbox'][2], p['bbox'][1]+p['bbox'][3]
	        ))
	
	with open(store_filename, 'w') as f:
	    for ann in imdb_lines:
	        f.write(ann + '\n')
	
	print('Stored:', store_filename)
	
	return store_filename



def annotations_to_imageset(annotations_filename, store_filename=None):
	"""
	Creates an ImageSet from an annotations file.
	The ImageSet has the format: image_label image_id (str, int)
	
	Parameters
	----------
	annotations_filename : str
		Path to annotations file. Must be a .txt with the format:
			Row format: image_file_path box1 box2 ... boxN \n
			Box format: x_min,y_min,x_max,y_max,class_id (no space).
	store_filename : str, optional
		Path where to store the create ImageSet

	Returns
	-------
	store_filename : str
		Path to the new ImageSet file.
	"""

	if store_filename is None:
		store_filename = annotations_filename.replace('.txt', '_image_set.txt')
		
	if os.path.isfile(store_filename): 
		print('ImageSet already computed:', store_filename)
		return store_filename
	print('Computing ImageSet')
		
	with open(annotations_filename, 'r') as f: annotations = f.read().splitlines()
	
	# image_set = [ '{} {}'.format(ann.split()[0][:-5], i+1) for i,ann in enumerate(annotations) ]
	image_set = [ '{} {}'.format('/'.join(ann.split()[0][:-5].split('/')[-2:]), i+1) for i,ann in enumerate(annotations) ]
	image_set = sorted(image_set)
	

	with open(store_filename, 'w') as f:
	    for s in image_set:
	        f.write(s + '\n')
	
	print('Stored:', store_filename)
	
	return store_filename



def image_set_to_motion_file(motion_iou_file_orig, imageset_filename_orig, imageset_filename_dest, motion_iou_dest_filename=None):
	"""
	Given the original ImageNet motion file (.mat), its original ImageSet 
	and a sub set of the original ImageSet, parses the motion file to fit the 
	new ImageSet

	Parameters
	----------
	motion_iou_file_orig : .mat
		Path to the original .mat file that contains the stats of each object related to
		its motion.
	imageset_filename_orig : str
		Path to the full ImagenetVid ImageSet.
	imageset_filename_dest : str
		Path to the new ImageSet (subset of ImagenetVid).
	motion_iou_dest_filename : str , optional
		Path where the new motion file will be stored.

	Returns
	-------
	motion_iou_dest_filename : str
		Path where the new motion file is stored.
	"""
	
	if motion_iou_dest_filename is None:
		motion_iou_dest_filename = imageset_filename_dest.replace('_image_set.txt', '_motion_iou.mat')
		
	if os.path.isfile(motion_iou_dest_filename): 
		print('Motion File already computed:', motion_iou_dest_filename)
		return motion_iou_dest_filename
	print('Computing Motion File:', motion_iou_dest_filename)
		
	motion_iou = sio.loadmat(motion_iou_file_orig)['motion_iou']
	
	with open(imageset_filename_orig, 'r') as f: image_set_orig = f.read().splitlines()
	image_set_orig = [ s.split()[0] for s in image_set_orig ]
	
	with open(imageset_filename_dest, 'r') as f: image_set_dest = f.read().splitlines()
	image_set_dest = [ s.split()[0] for s in image_set_dest ]
	
#	inds = [ s in image_set_dest for s in tqdm(image_set_orig) ]
	inds = [ s in image_set_dest for s in image_set_orig ]
	# motion_iou_dest = np.expand_dims(motion_iou[inds], axis=1)
	motion_iou_dest = motion_iou[inds]
	

	sio.savemat(motion_iou_dest_filename, {'motion_iou': motion_iou_dest})
	
	print('Stored:', motion_iou_dest_filename)
	
	return motion_iou_dest_filename



def print_mAP(ap_data, motion_ranges, area_ranges):
	for motion_index, motion_range in enumerate(motion_ranges):
	    for area_index, area_range in enumerate(area_ranges):
	        print('=================================================')
	        print('motion [{0:.1f} {1:.1f}], area [{2} {3} {4} {5}]'.format(
	            motion_range[0], motion_range[1], np.sqrt(area_range[0]), np.sqrt(area_range[0]),
	            np.sqrt(area_range[1]), np.sqrt(area_range[1])))
	        print('Mean AP@0.5 = {:.4f}'.format(np.mean(
	            [ap_data[motion_index][area_index][i] for i in range(len(ap_data[motion_index][area_index])) if
	             ap_data[motion_index][area_index][i] >= 0])))


def parse_ap_data(ap_data):
	stats = {
				'mAP_total': np.mean([ap_data[0][0][i] for i in range(len(ap_data[0][0])) if ap_data[0][0][i] >= 0]),
				'mAP_slow': np.mean([ap_data[3][0][i] for i in range(len(ap_data[3][0])) if ap_data[3][0][i] >= 0]),
				'mAP_medium': np.mean([ap_data[2][0][i] for i in range(len(ap_data[2][0])) if ap_data[2][0][i] >= 0]),
				'mAP_fast': np.mean([ap_data[1][0][i] for i in range(len(ap_data[1][0])) if ap_data[1][0][i] >= 0]),
			}
	return stats


