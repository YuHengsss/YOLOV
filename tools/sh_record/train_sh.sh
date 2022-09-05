conda activate yolov
cd yh/YOLOV
tensorboard --logdir ./YOLOX_outputs

sudo mount /dev/nvme0n1p1 /media/tuf/ssd/
sudo mount /dev/sdb /media/ssd/

python tools/new_train.py -f exps/vp_msim/yolovs_thresh_msim.py -c YOLOX_outputs/yolovs_thresh_msim/latest_ckpt.pth --resume
python tools/new_train.py -f exps/vp_msim/yolovx_thresh_msim.py -c weights/yoloxx_bn.pth

python tools/new_train.py -f exps/vp_trans/yolovs_ptrans_aug.py -c weights/yoloxs_vid_ori2.pth

python tools/new_train.py -f exps/vp_msim/yolovx_thresh_msim.py -c weights/yoloxx_bn.pth

python tools/new_train.py -f exps/test_dir/test_2head.py -c weights/yoloxs_vid_ori2.pth

python tools/new_train.py -f exps/vp_aug/yolovxp_trans4_aug.py -c ../YOLOX/YOLOX_outputs/yolovpx_trans4/latest_ckpt.pth

python tools/new_train.py -f exps/vp_trans/yolovps_msa_drop.py -c weights/yolovps_msa.pth

python tools/new_train.py -f exps/vp_trans/yolovxp_trans4_aug_drop.py -c weights/yolovxp_trans4_aug.pth


python tools/new_train.py -f exps/vp_trans/yolovxp_trans4_aug_and_drop1.py -c weights/yolovxp_trans4_aug.pth  #verify drop & l1 loss -2080ti

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/vp_trans/yolovxp_trans4_aug_drop.py -c weights/yolovxp_trans4_aug.pth  #verify close strong aug's best ap -3060

python tools/new_train.py -f exps/vp_aug/yolovsp_t4aug_scratch.py -c weights/yoloxs_vid_ori2.pth  #verify from scratch small -3060

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/vp_trans/yolovxp_trans4_auged.py -c weights/yolovxp_trans4_auged.pth #add l1 and no mosaic from yolovxp_trans4_aug.pth -3060

python tools/new_train.py -f exps/vp_aug/yolovxp_t4aug_drop1.py -c YOLOX_outputs/yolovxp_trans4_aug/last_mosaic_epoch_ckpt.pth #-2080ti verify drop rate in aug with/wo l1 loss

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/test_dir/yolovxpt4_test_auged.py -c weights/yolovxp_trans4_aug.pth ##add l1 and no mosaic from yolovxp_trans4_aug.pth with small lr -3060

python tools/new_train.py -f exps/test_dir/yolovxpt4_test_l1.py -c weights/yolovxp_trans4_aug.pth ##close l1 and no mosaic from yolovxp_trans4_aug.pth with small lr -2080ti

python tools/new_train.py -f exps/test_dir/yolovspt4_after_scratch.py -c YOLOX_outputs/yolovsp_t4aug_scratch/latest_ckpt.pth

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/test_dir/test_schedule_s.py -c weights/yoloxs_vid_ori2.pth #-3060

python tools/new_train.py -f exps/test_dir/test_schedule_x.py -c weights/yoloxx_bn.pth #-2080Ti

python tools/new_train.py -f exps/test_dir/test_schedule_s_drop1.py -c weights/yoloxs_vid_ori2.pth #-3060

python tools/new_train.py -f exps/test_dir/test_schedule_s_v2.py -c weights/yoloxs_vid_ori2.pth #-3060

python tools/new_train.py -f exps/test_dir/test_schedule_x_v2.py -c weights/yoloxx_bn.pth #i9

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/test_dir/test_schedule_s_v3.py -c weights/warmed_s_v2.pth  #-3060

python tools/new_train.py -f exps/test_dir/test_schedule_s_v4.py -c weights/warmed_s_v2.pth --resume #-3060

python tools/new_train.py -f exps/vp_trans/yolovxp_t4d2.py -c weights/yolovxpt4_845.pth #-i9

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/test_dir/test_msa_valina.py -c weights/yoloxs_vid_ori2.pth #-3060

python tools/new_train.py -f exps/test_dir/final_schedule_l.py -c weights/yolox_l_vid.pth #-3060

python tools/new_train.py -f exps/yolov/final_schedule_x2_v2.py -c weights/yoloxx_vid805.pth #-3060


-------------------------------------TODO-----------------------------------------
#figure out mosaic l1 loss in code

#verify from scratch xlarge

#global mosaic aug

#
python tools/new_eval.py -f exps/test_dir/final_schedule_x.py  -c weights/yolov_x.pth #0.7

-------------------------------------Ablation-----------------------------------------
python tools/new_train.py -f exps/ablation/raw_trans.py -c weights/yoloxs_vid_ori2.pth #-3060 5000/10753

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ablation/raw_trans_25.py -c weights/yoloxs_vid_ori2.pth #-3060 5000/10753


python tools/new_train.py -f exps/ablation/raw_trans_scaled_linear.py -c weights/yoloxs_vid_ori2.pth #-i9 5000/10753

python tools/new_train.py -f exps/ablation/yolovs_nr_default.py -c weights/yoloxs_vid_ori2.pth #-3060

python tools/new_train.py -f exps/ablation/test_thresh6.py -c weights/yoloxs_vid_ori2.pth --tsize 576 --fp16

CUDA_VISIBLE_DEVICES=1  python tools/new_train.py -f exps/ablation/test_thresh9.py -c weights/yoloxs_vid_ori2.pth --tsize 576 --fp16
-------------------------------------PPYOLOE-----------------------------------------
python tools/new_train.py -f exps/ppyoloe/ppv_l_754.py -c weights/ppyoloe/ppyoloel_vid.pth

python tools/new_train.py -f exps/ppyoloe/ppv_l_final2.py -c weights/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppv_l_wo2.py -c YOLOX_outputs/ppyoloel/best_ckpt.pth --fp16

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppv_l_woextra.py -c weights/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train.py -f exps/ppyoloe/ppyoloel.py -c weights/ppyoloe_l.pth -d 4 -b 32

python tools/new_train.py -f exps/ppyoloe/ppv_l_ese.py -c weights/ppyoloe/ppyoloel_vid773.pth --fp16

python tools/train.py -f exps/ppyoloe/ppyoloel_xhead.py -c weights/ppyoloe/ppyoloel_vid773.pth --fp16 -b 16

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppv_l_msim.py -c weights/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train.py -f exps/ppyoloe/ppyoloes.py -c weights/ppyoloe_s_coco.pth -d 4 -b 64 --fp16

python tools/new_eval.py -f exps/ppyoloe/ppv_l_50.py -c YOLOX_outputs/ppv_l_50/best_ckpt.pth --tsize 512

python tools/train.py -f exps/ppyoloe/ppyoloex.py -c weights/ppyoloex.pth -d 4 -b 24 --fp16

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppv_l_vallina.py -c weights/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppv_s25_backbone.py -c weights/ppyoloes_backbone_vid.pth --fp16 --tsize 512

python tools/train.py -f exps/ppyoloe/ppyoloes.py -c weights/ppyoloes_backbone_vid.pth --fp16 -b 24 #尝试把 from backbone fintue 的 ppyoloes 训练到和使用coco pretrain 一样的精度

python tools/new_train -f exps/ppyoloe/ppv_s_test_freaze -c weights/ppyoloes_vid_695.pth --fp16

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppvl_75_75_750.py -c weights/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppvs_75_75_750.py -c weights/ppyoloes_vid_695.pth

python tools/train.py -f exps/ppyoloe/ppvl_xhead_75_750.py -c weights/ppyoloe/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppvl_20_20_750.py -c weights/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/ppyoloe/ppyoloex.py -c weights/ppyoloex.pth -d 4 -b 8

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/ppyoloe/ppvl_20_20_750_fbn.py -c weights/ppyoloel_vid769.pth

CUDA_VISIBLE_DEVICES=0 python tools/new_train.py -f exps/ppyoloe/ppv_l_final_reple.py -c weights/ppyoloel_vid769.pth

-------------------------------------FCOS-----------------------------------------
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train.py -f exps/fcos/fcos_r50.py -c weights/fcos_r50_fpn_2x.pth -d 4 -b 24 --fp16
CUDA_VISIBLE_DEVICES=1 python tools/train.py -f exps/fcos/fcos_r50.py -c YOLOX_outputs/fcos_r50/latest_ckpt.pth --resume -d 1 -b 6 --fp16

python tools/new_train.py -f exps/fcos/fr50_v_woscore.py -c weights/fcos_r50.pth #3060

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/fcos/fr50_v.py -c weights/fcos_r50.pth

python tools/new_train.py -f exps/fcos/fr50_v_msim.py -c weights/fcos_r50.pth #my sever

CUDA_VISIBLE_DEVICES=1 python tools/new_train.py -f exps/fcos/fr50_v.py -c weights/fcos_r50.pth #3000 samples

python tools/new_train.py -f exps/fcos/fr50_v_freazeBN.py -c weights/fcos_r50.pth
python tools/new_train.py -f exps/fcos/fr50_v_freazeBN.py -c YOLOX_outputs/fr50_v_freazeBN/latest_ckpt.pth --resume

python tools/new_train.py -f exps/fcos/fr50_v_freazeBN_ncls.py -c weights/fcos_r50.pth --tsize 512 --fp16

python tools/train.py -f exps/fcos/fcos_r50_3strides.py -c weights/fcos_r50.pth -d 1 -b 8 --fp16

python tools/new_train.py -f exps/fcos/fr50_v_3strides.py -c YOLOX_outputs/fcos_r50_3strides/best_ckpt.pth --fp16 --tsize 512
-------------------------------------New YOLOV-----------------------------------------
python tools/train.py -f exps/example/custom/yoloxx.py -c weights/yolox_x.pth -d 4 -b 32 --fp16


python tools/train.py -f exps/example/custom/yoloxx.py -c YOLOX_outputs/yoloxx/best_ckpt.pth -d 4 -b 24 --fp16 --resume

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train.py -f exps/yolov/yoloxx_p6.py -c weights/yolox_x.pth -d 4 -b 16 --fp16
-------------------------------------YOLOV Epic-----------------------------------------
CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train.py -f exps/epic/yoloxx_epic.py -c weights/yolox_x.pth -d 4 -b 32 --fp16

CUDA_VISIBLE_DEVICES=4,5, python tools/train.py -f exps/epic/yoloxs_epic.py -c YOLOX_outputs/yoloxs_epic/latest_ckpt.pth -d 2 -b 32 --resume

CUDA_VISIBLE_DEVICES=3 python tools/train_epic.py -f exps/epic/yolovx_epic.py -c YOLOX_outputs/yoloxx_epic_more_iter/latest_ckpt.pth --fp16

CUDA_VISIBLE_DEVICES=4,5 python tools/train.py -f exps/epic/yoloxx_epic_more_iter.py -c YOLOX_outputs/yoloxx_epic_more_iter/latest_ckpt -d 2 -b 12 --resume --fp16

CUDA_VISIBLE_DEVICES=3 python tools/train_epic.py -f exps/epic/yolovx_epic.py -c YOLOX_outputs/yoloxx_epic_more_iter/epoch_18_ckpt.pth --fp16

CUDA_VISIBLE_DEVICES=0 python tools/train_epic.py -f exps/epic/yolovx_epic.py -c YOLOX_outputs/yolovx_epic/latest_ckpt.pth --fp16 -b 32 --gframe 32


CUDA_VISIBLE_DEVICES=2,3 python tools/train.py -f exps/epic/yoloxx_epic_hw_iter.py -c YOLOX_outputs/yoloxx_epic_hw_iter/latest_ckpt -d 2 -b 12 --resume --fp16

CUDA_VISIBLE_DEVICES=6,7 python tools/train.py -f exps/epic/yoloxx_epic_hw_scratch.py -c weights/yolox_x.pth -d 2 -b 12 --resume --fp16


CUDA_VISIBLE_DEVICES=6,7 python tools/train.py -f exps/epic/yoloxx_epic_640.py -c YOLOX_outputs/yoloxx_epic_hw_scratch/latest_ckpt.pth -d 2 -b 12 --fp16

CUDA_VISIBLE_DEVICES=0,1,2 python tools/train.py -f exps/epic/yoloxx_epic_v2.py -c weights/yolox_x.pth -d 3 -b 12 --fp16

-------------------------------------YOLOV args-----------------------------------------
CUDA_VISIBLE_DEVICES=0 python tools/train.py -f exps/yolo_arg/yoloxx_arg.py -c weights/yolox_x.pth -d 1 -b 6 --fp16 --cache

python tools/train_vid.py -f exps/yolo_arg/yolovs_arg.py -c weights/yoloxs_arg_960.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/yolo_arg/yoloxx_arg.py -c weights/yolox_x.pth -d 4 -b 8 --fp16 --cache


CUDA_VISIBLE_DEVICES=0 python tools/train_vid.py -f exps/yolo_arg/yolovs_arg_nc.py -c YOLOX_outputs/yoloxs_arg/best_ckpt.pth

CUDA_VISIBLE_DEVICES=2 python tools/train_vid.py -f exps/yolo_arg/yolovx_arg_nc.py -c YOLOX_outputs/yoloxx_arg/best_ckpt.pth  --fp16

CUDA_VISIBLE_DEVICES=0 python tools/train_vid.py -f exps/yolo_arg/yolovx_arg_nc85.py -c YOLOX_outputs/yoloxx_arg/best_ckpt.pth

CUDA_VISIBLE_DEVICES=1 python tools/train_vid.py -f exps/yolo_arg/yolovx_arg_nc90.py -c YOLOX_outputs/yoloxx_arg/best_ckpt.pth

CUDA_VISIBLE_DEVICES=2 python tools/train_vid.py -f exps/yolo_arg/yolovs_arg_nc85_fbn.py -c YOLOX_outputs/yoloxs_arg/best_ckpt.pth

CUDA_VISIBLE_DEVICES=2 python tools/train_vid.py -f exps/yolo_arg/yolovx_arg_nc85_fbn_e2.py -c YOLOX_outputs/yoloxx_arg/epoch_2_ckpt.pth

-------------------------------------YOLOV ovis-----------------------------------------

CUDA_VISIBLE_DEVICES=0 python tools/train.py -f exps/yolo_ovis/yoloxx_ovis.py -c weights/yolox_x.pth -d 1 -b 2 --fp16

CUDA_VISIBLE_DEVICES=0,1 python tools/train.py -f exps/yolo_ovis/yoloxs_ovis.py -c weights/yolox_s.pth -d 2 -b 12 --fp16

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/yolo_ovis/yoloxx_ovis.py -c weights/yolox_x.pth -d 4 -b 8 --fp16


CUDA_VISIBLE_DEVICES=1 python tools/train_vid.py -f exps/yolo_ovis/yolovs_ovis.py -c YOLOX_outputs/yoloxs_ovis/epoch_9_ckpt.pth

CUDA_VISIBLE_DEVICES=1 python tools/train_vid.py -f exps/yolo_ovis/yolovx_ovis.py -c YOLOX_outputs/yoloxx_ovis/epoch_9_ckpt.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/yolo_ovis/yoloxl_ovis.py -c weights/yolox_l.pth -d 4 -b 16 --fp16

CUDA_VISIBLE_DEVICES=1 python tools/train_vid.py -f exps/yolo_ovis/yolovl_ovis_75_75_750.py -c YOLOX_outputs/yoloxl_ovis/epoch_9_ckpt.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/yolo_ovis/yoloxx_ovis.py -c YOLOX_outputs/yoloxx_ovis/latest_ckpt.pth -d 4 -b 8 --fp16 --resume

CUDA_VISIBLE_DEVICES=0,1 python tools/train.py -f exps/yolo_ovis/yoloxl_ovis_b8.py -c weights/yolox_l.pth -d 2 -b 8 --fp16

CUDA_VISIBLE_DEVICES=2 python tools/train_vid.py -f exps/yolo_ovis/yolovs_ovis_75_75_750.py -c YOLOX_outputs/yoloxs_ovis/epoch_9_ckpt.pth

CUDA_VISIBLE_DEVICES=3 python tools/train_vid.py -f exps/yolo_ovis/yolovl_ovis_75_75_750copy.py -c YOLOX_outputs/yoloxl_ovis/epoch_9_ckpt.pth

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f exps/ppyoloe/ppyoloex.py -c YOLOX_outputs/ppyoloex/latest_ckpt.pth -d 4 -b 12 --fp16 --resume

CUDA_VISIBLE_DEVICES=0 python tools/train_vid.py -f exps/yolo_ovis/yolovs_ovis_75_750.py -c YOLOX_outputs/yoloxs_ovis/epoch_9_ckpt.pth

CUDA_VISIBLE_DEVICES=1 python tools/train_vid.py -f exps/yolo_ovis/yolovl_ovis_75_750.py -c YOLOX_outputs/yoloxl_ovis/epoch_9_ckpt.pth


