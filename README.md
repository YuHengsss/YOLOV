

# YOLOV for video object detection.

## Introduction
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov-making-still-image-object-detectors/video-object-detection-on-imagenet-vid)](https://paperswithcode.com/sota/video-object-detection-on-imagenet-vid?p=yolov-making-still-image-object-detectors)

YOLOV is a high perfomance video object detector.  Please refer to our [paper](https://arxiv.org/abs/2208.09686) on Arxiv for more details.

This repo is an implementation of PyTorch version YOLOV based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

<img src="assets/comparsion.jpg" width="500" >


## Main result


| Model                                                                                                               | size | mAP@50<sup>val<br> | Speed 2080Ti(batch size=1)<br>(ms) |                                           weights                                            |
|---------------------------------------------------------------------------------------------------------------------|:----:|:------------------:|:----------------------------------:|:--------------------------------------------------------------------------------------------:|
| YOLOX-s                                                                                                             | 576  |        69.5        |                9.4                 |                                         [google](https://drive.google.com/file/d/1n8wkByqpHdrGy6z9fsoZpBtTa0I3JOcG/view?usp=sharing)                                          |
| YOLOX-l                                                                                                             | 576  |        76.1        |                14.8                |                                         [google](https://drive.google.com/file/d/1rikaPCAHBBIugYUZYV1buyOIRG8xvGKB/view?usp=sharing)                                          |
| YOLOX-x                                                                                                             | 576  |        77.8        |                20.4                |                                         [google](https://drive.google.com/file/d/1OH3hGj7RMfcinMKPESbfI7C5y_RrA3aF/view?usp=sharing)                                          |
| YOLOV-s                                                                                                             | 576  |        77.3        |                11.3                | [google](https://drive.google.com/file/d/12X4dQw45aXVYgJjKAAAPk409FO3xValW/view?usp=sharing) |
| YOLOV-l                                                                                                             | 576  |        83.6        |                16.4                | [google](https://drive.google.com/file/d/1qZ-3iPDlYx1OKe6zz_-n42ceijo_Ntx6/view?usp=sharing) |
| YOLOV-x                                                                                                             | 576  |        85.5        |                22.7                | [google](https://drive.google.com/file/d/1OIozS-D9wbWA9pDFl5xoFw6XqEcYtzsJ/view?usp=sharing) |
| YOLOV-x + [post](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection) | 576  |        87.5        |                 -                  |                                              -                                               |


## TODO
Finish Swin-Transformer based experiments.

## Quick Start

<details>
<summary>Installation</summary>

Install YOLOV from source.
```shell
git clone git@github.com:YuHengsss/YOLOV.git
cd YOLOV
```

Create conda env.
```shell
conda create -n yolov python=3.7

conda activate yolov

pip install -r requirements.txt

pip3 install -v -e .
```
</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained weights.

Step2. Run yolov demos. For example:

```shell
python tools/vid_demo.py -f [path to your yolov exp files] -c [path to your yolov weights] --path /path/to/your/video --conf 0.25 --nms 0.5 --tsize 576 --save_result 
```
For online mode, exampled with yolov_l, you can run:

```shell
python tools/yolov_demo_online.py -f ./exp/yolov/yolov_l_online.py -c [path to your weights] --path /path/to/your/video --conf 0.25 --nms 0.5 --tsize 576 --save_result 
```
For yolox models, please use python tools/demo.py for inferencing.
</details>

<details>
<summary>Reproduce our results on VID</summary>

Step1. Download datasets and weights:

Download ILSVRC2015 DET and ILSVRC2015 VID dataset from [IMAGENET](https://image-net.org/challenges/LSVRC/2015/2015-downloads) and organise them as follows:

```shell
path to your datasets/ILSVRC2015/
path to your datasets/ILSVRC/
```

Download our COCO-style annotations for [training](https://drive.google.com/file/d/1HhE4OAcc--CpjUj69JCRXzMvIRsR4ymM/view?usp=sharing) and [video sequences](https://drive.google.com/file/d/1vJs8rLl_2oZOWCMJtk3a9ZJmdNn8cu-G/view?usp=sharing). Then, put them in these two directories:
```shell
YOLOV/annotations/vid_train_coco.json
YOLOV/yolox/data/dataset/train_seq.npy
```

Change the data_dir in exp files to [path to your datasets] and Download our weights.

Step2. Generate predictions and convert them to IMDB style for evaluation.

```shell
python tools/val_to_imdb.py -f exps/yolov/yolov_x.py -c path to your weights/yolov_x.pth --fp16 --output_dir ./yolov_x.pkl
```
Evaluation process:
```shell
python tools/REPPM.py --repp_cfg ./tools/yolo_repp_cfg.json --predictions_file ./yolov_x.pkl --evaluate --annotations_filename ./annotations/annotations_val_ILSVRC.txt --path_dataset [path to your dataset] --store_imdb --store_coco  (--post)
```
(--post) indicates involving post-processing method. Then you will get:
```shell
{'mAP_total': 0.8758871720817065, 'mAP_slow': 0.9059275666099181, 'mAP_medium': 0.8691557352372217, 'mAP_fast': 0.7459511040452989}
```

  
**Training example**
```shell
python tools/vid_train.py -f exps/yolov/yolov_s.py -c weights/yoloxs_vid.pth --fp16
```
**Roughly testing**
```shell
python tools/vid_eval.py -f exps/yolov/yolov_s.py -c weights/yolov_s.pth --tnum 500 --fp16
```
tnum indicates testing sequence number.
</details>


## Annotation format in YOLOV
<details>
  
**Training base detector**
The train_coco.json is a COCO format annotation file. When trainig the base detector on your own dataset, try to convert the annotation to COCO format.

**Training YOLOV**
The train_seq.npy and val_seq.npy files are numpy arrays of lists. They can be loaded using the following command:
```shell
numpy.load('./yolox/data/datasets/train_seq.npy',allow_pickle=True)
```
Each list contains the paths to all images in a video. The specific annotations(xml annotation in VID dataset) are loaded via these image paths, refer to https://github.com/YuHengsss/YOLOV/blob/f5a57ddea2f3660875d6d75fc5fa2ddbb95028a7/yolox/data/datasets/vid.py#L125 for more details.
</details>

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection)
</details>

## Cite YOLOV
If YOLOV is helpful for your research, please cite the following paper:

```latex
@article{shi2022yolov,
  title={YOLOV: Making Still Image Object Detectors Great at Video Object Detection},
  author={Shi, Yuheng and Wang, Naiyan and Guo, Xiaojie},
  journal={arXiv preprint arXiv:2208.09686},
  year={2022}
}
```
