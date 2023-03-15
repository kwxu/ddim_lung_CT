#!/bin/bash

PYTHON_EXE=/home/xuk9/anaconda3/envs/DDIM/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/ddim_lung_CT

EXP_PATH=/local_storage/xuk9/Projects/DDIM_lung_CT/DDIM_classroom/exp1

${PYTHON_EXE} ${SRC_ROOT}/main.py \
  --config ${SRC_ROOT}/configs/classroom.yml \
  --exp ${EXP_PATH} \
  --doc ddim_example_classroom \
  --ni