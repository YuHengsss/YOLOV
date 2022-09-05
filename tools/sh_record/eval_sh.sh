conda activate yolov
cd yh/YOLOV
python tools/new_eval.py -f exps/vp_trans/yolovx_ptrans_qv.py -c YOLOX_outputs/yolovx_ptrans_qv/latest_ckpt.pth --tsize 576

python tools/new_eval.py -f exps/vp_msim/yolovx_thresh_msim.py -c YOLOX_outputs/yolovx_thresh_msim/latest_ckpt.pth --tsize 576



python tools/new_eval.py -f exps/vp_msim/yolovx_thresh_msim.py -c YOLOX_outputs/yolovx_thresh_msim/latest_ckpt.pth --tsize 576

python tools/new_train.py -f exps/vp_msim/yolovx_thresh_msim.py -c YOLOX_outputs/yolovxp_aug_msim/latest_ckpt.pth --tsize 512

python tools/new_eval.py -f exps/example/custom/yolovx_thresh_2head.py -c weights/833.pth

CUDA_VISIBLE_DEVICES=1 python tools/new_eval.py -f exps/vp_trans/yolovxp_trans4_aug_drop.py -c weights/yolovxp_trans4_aug.pth

python tools/new_eval.py -f exps/vp_aug/yolovxp_trans4_aug.py -c ./YOLOX_outputs/yolovxp_trans4_aug/best_ckpt.pth

python tools/new_eval.py -f exps/test_dir/test_schedule_x_v2.py -c ./weights/test_schedule_x.pth


python tools/val_to_repp_online.py -f exps/test_dir/test_l_online.py -c weights/yolov_l.pth --output_dir /home/hdr/yh/YOLOV/YOLOX_outputs/yolov_l_online.pckl --path /home/hdr/yh/YOLOV/yolox/data/datasets/val_seq.npy

CUDA_VISIBLE_DEVICES=1 python tools/val_to_repp_online.py -f exps/test_dir/test_s_online.py -c weights/yolov_s.pth --output_dir /home/hdr/yh/YOLOV/YOLOX_outputs/yolov_s_online.pckl --path /home/hdr/yh/YOLOV/yolox/data/datasets/val_seq.npy



# Validate acc of ppyoloe in terms of sample number

CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py -f exps/ppyoloe/ppv_l_final2.py -c weights/ppv_l_final2.pth --tsize 512 #30

CUDA_VISIBLE_DEVICES=1 python tools/new_eval.py -f exps/ppyoloe/ppv_l_final2.py -c weights/ppv_l_final2.pth --tsize 512 #50

CUDA_VISIBLE_DEVICES=2 python tools/new_eval.py -f exps/ppyoloe/ppv_l_final2.py -c weights/ppv_l_final2.pth --tsize 512 #75

python tools/new_eval.py -f exps/test_dir/test_schedule_x_v2.py -c weights/yolov_x.pth --tsize 576

CUDA_VISIBLE_DEVICES=5,6,7 python tools/eval.py -f exps/epic/yoloxx_epic_more_iter.py -c YOLOX_outputs/yoloxx_epic_more_iter/latest_ckpt.pth -d 3 -b 36

python tools/eval_epic.py eval -f exps/epic/yoloxx_epic_more_iter.py -c YOLOX_outputs/yoloxx_epic_more_iter/best_ckpt.pth --path ./annotations/Epic_s2_test.npy

CUDA_VISIBLE_DEVICES=3 python tools/eval_epic_vid.py eval -f exps/epic/yolovx_epic.py -c YOLOX_outputs/yolovx_epic/latest_ckpt.pth --path ./annotations/Epic_vid_s2.npy

CUDA_VISIBLE_DEVICES=3 python tools/eval_epic_vid.py eval -f exps/epic/yolovx_epic.py -c YOLOX_outputs/yolovx_epic/latest_ckpt.pth --path ./annotations/Epic_vid_s1.npy

CUDA_VISIBLE_DEVICES=3 python tools/eval_epic.py eval -f exps/epic/yoloxx_epic_hw_iter.py -c ./YOLOX_outputs/yoloxx_epic_hw_iter/best_ckpt.pth --path ./annotations/Epic_s2_test.npy

CUDA_VISIBLE_DEVICES=1 python tools/eval_epic.py eval -f exps/epic/yoloxx_epic_hw_iter.py -c ./YOLOX_outputs/yoloxx_epic_hw_iter/best_ckpt.pth --path ./annotations/Epic_s1_test.npy


CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f exps/epic/yolovx_epic_hw_iter.py -c YOLOX_outputs/yolovx_epic_hw_iter/best_ckpt.pth -d 1 -b 16 --fp16


CUDA_VISIBLE_DEVICES=3 python tools/eval_epic_vid.py eval -f exps/epic/yolovx_epic_hw_iter.py -c YOLOX_outputs/yolovx_epic_hw_iter/latest_ckpt.pth --path ./annotations/Epic_vid_s1.npy

CUDA_VISIBLE_DEVICES=3 python tools/eval_epic_vid.py eval -f exps/epic/yolovx_epic_hw_iter.py -c YOLOX_outputs/yolovx_epic_hw_iter/latest_ckpt.pth --path ./annotations/Epic_vid_s2.npy


CUDA_VISIBLE_DEVICES=0 python tools/eval_epic.py eval -f exps/epic/yoloxx_epic_v2.py -c YOLOX_outputs/yoloxx_epic_v2/best_ckpt.pth --path ./annotations/Epic_s2_test.npy

CUDA_VISIBLE_DEVICES=1 python tools/eval_epic.py eval -f exps/epic/yoloxx_epic_v2.py -c YOLOX_outputs/yoloxx_epic_v2/best_ckpt.pth --path ./annotations/Epic_s1_test.npy

CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py  -f exps/yolo_arg/yolovx_arg_nc.py -c YOLOX_outputs/yolovx_arg_nc85/best_ckpt.pth

CUDA_VISIBLE_DEVICES=2 python tools/new_eval.py  -f exps/yolo_arg/yolovx_arg_nc90.py -c YOLOX_outputs/yolovx_arg_nc90/best_ckpt.pth --gframe 0 --lframe 32

CUDA_VISIBLE_DEVICES=2 python tools/eval_arg_vid.py eval -f exps/yolo_arg/yolovs_arg_nc85.py -c YOLOX_outputs/yolovs_arg_nc85/best_ckpt.pth --path /opt/dataset/Argoverse-1.1/annotations/val.json --data_dir /opt/dataset/Argoverse-1.1 --fp16

CUDA_VISIBLE_DEVICES=3 python tools/eval_arg_vid.py eval -f exps/yolo_arg/yolovx_arg_nc85.py -c YOLOX_outputs/yolovx_arg_nc85/best_ckpt.pth --path /opt/dataset/Argoverse-1.1/annotations/val.json --data_dir /opt/dataset/Argoverse-1.1 --fp16

tools/eval.py -f exps/yolo_arg/yolovs_arg_test.py -c weights/yolovs_arg_nc85_fbn.pth -b 32 --fp16

CUDA_VISIBLE_DEVICES=3 python tools/eval_ovis_vid.py eval -f exps/yolo_ovis/yolovx_ovis.py -c YOLOX_outputs/yolovx_ovis/latest_ckpt.pth --save_dir yolovx_100.json

CUDA_VISIBLE_DEVICES=3 python tools/eval_ovis_vid.py eval -f exps/yolo_ovis/yolovx_ovis_75_75_750.py -c YOLOX_outputs/yolovx_ovis_75_75_750/latest_ckpt.pth --save_dir yolovx_7575750.json

CUDA_VISIBLE_DEVICES=2 python tools/eval_ovis_vid.py eval -f exps/yolo_ovis/yolovl_ovis_75_75_750copy.py -c YOLOX_outputs/yolovl_ovis_75_75_750copy/latest_ckpt.pth --save_dir yolovl_ovis_75_75_750copy.json

conda activate yolov
cd yh/YOLOV
CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py  -f exps/test_dir/test_thresh.py -c weights/yolov_s.pth --tsize 576 --fp16 #0.2

CUDA_VISIBLE_DEVICES=1 python tools/new_eval.py  -f exps/test_dir/test_thresh.py -c weights/yolov_s.pth --tsize 576 --fp16 #0.3

CUDA_VISIBLE_DEVICES=2 python tools/new_eval.py  -f exps/test_dir/test_thresh.py -c weights/yolov_s.pth --tsize 576 --fp16 #0.4

CUDA_VISIBLE_DEVICES=3 python tools/new_eval.py  -f exps/test_dir/test_thresh.py -c weights/yolov_s.pth --tsize 576 --fp16 #0.5

CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py  -f exps/ablation/test_thresh6.py -c YOLOX_outputs/test_thresh6/latest_ckpt.pth --tsize 576 --fp16 #0.2

CUDA_VISIBLE_DEVICES=1 python tools/new_eval.py  -f exps/ablation/test_thresh9.py -c YOLOX_outputs/test_thresh9/latest_ckpt.pth --tsize 576 --fp16 #0.2



CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py  -f exps/ablation/test_thresh_woam0.py -c YOLOX_outputs/yolovs_our_msa_nrave/best_ckpt.pth --tsize 576 --fp16 #0.0 72.2

python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.1 72.8

CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.2 73.4

CUDA_VISIBLE_DEVICES=1 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.3 73.9

CUDA_VISIBLE_DEVICES=2 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.4 74.6

CUDA_VISIBLE_DEVICES=1 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c YOLOX_outputs/yolovs_our_msa_nrave/best_ckpt.pth --tsize 576 --fp16 #0.5 75.4

python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.6 75.9

CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.7 76.3

CUDA_VISIBLE_DEVICES=1 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.8 76.5

CUDA_VISIBLE_DEVICES=2 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.9 76.6

python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c weights/yolovs_woam.pth --tsize 576 --fp16 #0.95 76.6

CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py  -f exps/ablation/test_thresh_woam1.py -c YOLOX_outputs/yolovs_our_msa_nrave/best_ckpt.pth --tsize 576 --fp16 #0.9999 3


CUDA_VISIBLE_DEVICES=0 python tools/new_eval.py -f exps/ppyoloe/ppvs_20_20_750 -c YOLOX_outputs/ppvs_20_20_750/best_ckpt.pth --fp16 --tsize 576