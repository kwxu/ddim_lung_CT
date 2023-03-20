#!/bin/bash

PYTHON_EXE=/home/xuk9/anaconda3/envs/DDIM/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/ddim_lung_CT

EXP_PATH=/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/exp1

for ckpt_id in 35000 75000 150000 220000 300000
do
  ${PYTHON_EXE} ${SRC_ROOT}/main.py \
    --config ${SRC_ROOT}/configs/lung_ct_full_body.yml \
    --exp ${EXP_PATH} \
    --doc lung_ct_full_body \
    --sample \
    --sample_grid_plot \
    --ckpt_id ${ckpt_id} \
    --image_folder sample_grid_plot \
    --timesteps 50 \
    --eta 0 \
    --ni
done

