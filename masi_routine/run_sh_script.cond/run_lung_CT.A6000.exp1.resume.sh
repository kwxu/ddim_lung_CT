#!/bin/bash

#PYTHON_EXE=/home/local/VANDERBILT/xuk9/anaconda3/envs/DDIM/bin/python
PYTHON_EXE=/home/xuk9/anaconda3/envs/DDIM/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/ddim_lung_CT

EXP_PATH=/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT.cond/exp1

${PYTHON_EXE} ${SRC_ROOT}/main.py \
  --config ${SRC_ROOT}/configs.cond/lung_CT.fov_extension.exp1.yml \
  --exp ${EXP_PATH} \
  --doc lung_ct.cond \
  --resume_training \
  --ni