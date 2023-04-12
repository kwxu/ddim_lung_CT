import h5py
import os
import pandas as pd
from masi_routine.fov_extension_evaluation.eval_utils import InternalEvaluationUtilsRePaint, dict2namespace, InternalEvaluationData
import yaml


def run_inpainting():
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs.cond/lung_CT.fov_extension.exp1.yml'
    project_root = '/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT.cond/exp1'
    ckpt_path = os.path.join(project_root, 'models/ckpt_300000.pth')
    h5_dir = os.path.join(project_root, 'h5_internal_evaluation_v2')
    preview_dir = os.path.join(project_root, 'result_preview_png')

    with open(yml_config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    runner_utils = InternalEvaluationUtilsRePaint(config, ckpt_path, h5_dir)
    runner_utils.run_inference(
        preview_dir=preview_dir,
        # n_steps=50
        n_steps=250
    )


if __name__ == "__main__":
    run_inpainting()

