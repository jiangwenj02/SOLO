#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 ./tools/dist_train.sh configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py 8
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 ./tools/dist_train.sh configs/gangganpinghang/solov2_r50_fpn_gang_left_8gpu_3x.py 8
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 ./tools/dist_train.sh configs/adenomatous/solov2_r50_fpn_adenomatous_8gpu_3x.py 8
CUDA_VISIBLE_DEVICES=8 python tools/test_ins.py configs/gangganpinghang/solov2_r50_fpn_gang_left_8gpu_3x.py work_dirs/solov2_r50_fpn_gang_left_8gpu_3x/latest.pth --eval segm --out work_dirs/solov2_r50_fpn_gang_left_8gpu_3x/result.pkl --classwise True
CUDA_VISIBLE_DEVICES=8 python tools/test_ins_vis.py configs/gangganpinghang/solov2_r50_fpn_gang_left_8gpu_3x.py work_dirs/solov2_r50_fpn_gang_left_8gpu_3x/latest.pth --show --save_dir  /data0/zzhang/tmp/gangganpinghang
CUDA_VISIBLE_DEVICES=8 python tools/test_ins_vis.py configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth --show --save_dir  /data0/zzhang/tmp/cleaned_data
CUDA_VISIBLE_DEVICES=8 python tools/test_ins.py configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth --eval segm --out work_dirs/solov2_r50_fpn_8gpu_3x_poly/result.pkl --classwise True
#CUDA_VISIBLE_DEVICES=8,9,10,11 ./tools/dist_test.sh configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth 4 --eval segm --out work_dirs/solov2_r50_fpn_8gpu_3x_poly/result.pkl