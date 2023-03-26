#!/bin/bash

PYTHON_EXE=/home/local/VANDERBILT/xuk9/anaconda3/envs/DDIM/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/ddim_lung_CT

EXP_PATH=/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/exp3

${PYTHON_EXE} ${SRC_ROOT}/main.py \
  --config ${SRC_ROOT}/configs/lung_ct_full_body.exp3.narrow_range.yml \
  --exp ${EXP_PATH} \
  --doc lung_ct_full_body \
  --ni