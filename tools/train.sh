#!/bin/bash
export PYTHONPATH="/media/ubuntu/nvidia/wlq/co_work1_hilbert/HERO:$PYTHONPATH"

python train.py \
/media/ubuntu/nvidia/wlq/co_work1_hilbert/HERO/configs/_HERO/oriented-rcnn-le90_r50_fpn_6x_hrsc2016.py
python train.py \
/media/ubuntu/nvidia/wlq/co_work1_hilbert/HERO/configs/_HERO/HERO-le90_r50_fpn_6x_hrsc2016.py
#python train.py \
#/media/ubuntu/nvidia/wlq/part2/mmrotate/configs/_wlq/t2det/rtmdet_baseline/VEDAI/t2det_rtmdet_m-6x-vedai.py \
#--cfg-options model.bbox_head.ss_loss_start=0.2 model.bbox_head.loss_scale_ss.loss_weight=0 \
#--work-dir ./work_dirs/ablation/exp1/t2det_rtmdet_m-6x-vedai_divided-8_bs4_no2-ss0.2/
