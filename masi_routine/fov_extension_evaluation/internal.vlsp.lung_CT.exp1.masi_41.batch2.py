import h5py
import os
import pandas as pd
from masi_routine.fov_extension_evaluation.eval_utils import InternalEvaluationUtils, dict2namespace, InternalEvaluationData
import yaml

def run_repaint():
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs/lung_ct_full_body.exp1.yml'
    project_root = '/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT/exp1'
    ckpt_path = os.path.join(project_root, 'models/ckpt_360000.pth')
    # h5_dir = os.path.join(project_root, 'h5_internal_evaluation_v2.batch1')
    h5_dir = os.path.join(project_root, 'h5_internal_evaluation_v2.batch2')
    preview_dir = os.path.join(project_root, 'result_preview_png')

    with open(yml_config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    runner_utils = InternalEvaluationUtils(config, ckpt_path, h5_dir)
    runner_utils.run_inference(
        preview_dir=preview_dir
    )


if __name__ == "__main__":
    run_repaint()

