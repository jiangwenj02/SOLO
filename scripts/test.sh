#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=8 python tools/test.py configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth --eval segm --json_out work_dirs/solov2_r50_fpn_8gpu_3x_poly/result.json
#CUDA_VISIBLE_DEVICES=8,9,10,11 ./tools/dist_test.sh configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth 4 --eval segm --json_out work_dirs/solov2_r50_fpn_8gpu_3x_poly/result.json