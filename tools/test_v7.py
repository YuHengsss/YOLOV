import torch
from yolox.exp import get_exp
import pickle
import six


def copy(name, w, std):
    value2 = torch.Tensor(w)
    value = std[name]
    value.copy_(value2)
    std[name] = value


def covert_weights(option = 'yolov7l'):
    if option == 'yolov7l':
        ckpt_file = '../weights/yolov7_l_300e_coco.pdparams'
        output_ckpt = '../weights/yolov7l_backbone_fpn.pth'
        exp = get_exp('../exps/yolov7/yolov7l.py')
        model = exp.get_model()
        model.eval()
        model_std = model.state_dict()
        output = model(torch.ones([1, 3, 640, 640]))
        with open(ckpt_file, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        backbone_dic = {}
        fpn_dic = {}
        head_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic[key] = value
            elif 'neck' in key:
                fpn_dic[key] = value
            elif 'head' in key:
                head_dic[key] = value
            else:
                others[key] = value

        backbone_dic2 = {}
        fpn_dic2 = {}
        head_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic2[key] = value
            elif 'neck' in key:
                fpn_dic2[key] = value
            elif 'head' in key:
                head_dic2[key] = value
            else:
                others2[key] = value

        i_layer = 0
        for key in state_dict.keys():
            name2 = key
            w = state_dict[key]
            if 'StructuredToParameterName@@' in key:
                continue
            else:
                if 'head' in key:
                    break
                if 'tracked' in list(model_std.keys())[i_layer]:
                    i_layer += 1
                copy(list(model_std.keys())[i_layer], w, model_std)
                i_layer += 1
        ckpt_state = {
            "start_epoch": 0,
            "model": model.state_dict(),
            "optimizer": None,
        }
        torch.save(ckpt_state, output_ckpt)
    elif option == 'yolov7x':
        ckpt_file = '../weights/yolov7_x_300e_coco.pdparams'
        output_ckpt = '../weights/yolov7x_backbone_fpn.pth'
        exp = get_exp('../exps/yolov7/yolov7x.py')

        model = exp.get_model()
        model.eval()
        model_std = model.state_dict()
        output = model(torch.ones([1, 3, 640, 640]))
        with open(ckpt_file, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        backbone_dic = {}
        fpn_dic = {}
        head_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic[key] = value
            elif 'neck' in key:
                fpn_dic[key] = value
            elif 'head' in key:
                head_dic[key] = value
            else:
                others[key] = value

        backbone_dic2 = {}
        fpn_dic2 = {}
        head_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic2[key] = value
            elif 'neck' in key:
                fpn_dic2[key] = value
            elif 'head' in key:
                head_dic2[key] = value
            else:
                others2[key] = value

        i_layer = 0
        for key in state_dict.keys():
            name2 = key
            w = state_dict[key]
            if 'StructuredToParameterName@@' in key:
                continue
            else:
                if 'head' in key:
                    break
                if 'tracked' in list(model_std.keys())[i_layer]:
                    i_layer += 1
                copy(list(model_std.keys())[i_layer], w, model_std)
                i_layer += 1
        ckpt_state = {
            "start_epoch": 0,
            "model": model.state_dict(),
            "optimizer": None,
        }
        torch.save(ckpt_state, output_ckpt)
    elif option == 'yolov7p6w6':
        ckpt_file = '../weights/yolov7p6_w6_300e_coco.pdparams'
        output_ckpt = '../weights/yolov7p6w6_backbone_fpn.pth'
        exp = get_exp('../exps/yolov7/yolov7p6w6.py')

        model = exp.get_model()
        model.eval()
        model_std = model.state_dict()
        output = model(torch.ones([1, 3, 1280, 1280]))
        with open(ckpt_file, 'rb') as f:
            state_dict = pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        backbone_dic = {}
        fpn_dic = {}
        head_dic = {}
        others = {}
        for key, value in state_dict.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic[key] = value
            elif 'neck' in key:
                fpn_dic[key] = value
            elif 'head' in key:
                head_dic[key] = value
            else:
                others[key] = value

        backbone_dic2 = {}
        fpn_dic2 = {}
        head_dic2 = {}
        others2 = {}
        for key, value in model_std.items():
            if 'tracked' in key:
                continue
            if 'backbone' in key:
                backbone_dic2[key] = value
            elif 'neck' in key or 'fpn' in key:
                fpn_dic2[key] = value
            elif 'head' in key:
                head_dic2[key] = value
            else:
                others2[key] = value

        i_layer = 0
        for key in state_dict.keys():
            name2 = key
            w = state_dict[key]
            if 'StructuredToParameterName@@' in key:
                continue
            else:
                if 'head' in key:
                    break
                if 'tracked' in list(model_std.keys())[i_layer]:
                    i_layer += 1
                copy(list(model_std.keys())[i_layer], w, model_std)
                i_layer += 1
        ckpt_state = {
            "start_epoch": 0,
            "model": model.state_dict(),
            "optimizer": None,
        }
        torch.save(ckpt_state, output_ckpt)


if __name__ == "__main__":
    covert_weights('yolov7p6w6')
    print('Done')
