#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=8 python tools/test_ins_vis.py configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth --show --save_dir  /data0/zzhang/tmp/cleaned_data
CUDA_VISIBLE_DEVICES=8 python tools/test_ins.py configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth --eval segm --out work_dirs/solov2_r50_fpn_8gpu_3x_poly/result.pkl
#CUDA_VISIBLE_DEVICES=8,9,10,11 ./tools/dist_test.sh configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth 4 --eval segm --out work_dirs/solov2_r50_fpn_8gpu_3x_poly/result.pkl