import argparse
import os
import pickle
import six
import torch
from collections import OrderedDict

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint")
    parser.add_argument("-oc", "--out_ckpt", default=None, type=str, help="output checkpoint")
    return parser

def main(args):
    ckpt = torch.load(args.ckpt,map_location='cpu')
    converted_ckpt = {}
    converted_dict = OrderedDict()
    state_dict = ckpt['model']
    for key in state_dict.keys():
        name2 = key
        w = state_dict[key]
        if name2.split('.')[0] == 'head':continue
        if 'backbone.backbone.' not in name2:
            if name2.split('.')[0] != 'norm':
                name2 = 'backbone.backbone.' + name2
            else:
                name2 = name2.replace('norm.','backbone.backbone.norm3.')
        converted_dict[name2] = w
    converted_ckpt['model'] = converted_dict
    if args.out_ckpt is not None:
        torch.save(converted_ckpt, args.out_ckpt)
    return


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
