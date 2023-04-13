import h5py
import os
import pandas as pd
from masi_routine.fov_extension_evaluation.eval_utils import InternalEvaluationUtilsRePaint, dict2namespace, InternalEvaluationData
import yaml

def run_inpainting():
    source_h5_dir = '/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT.cond/exp2/h5_internal_evaluation_v2'
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs.cond/lung_CT.fov_extension.exp1.yml'
    ckpt_path = '/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT.cond/exp1/logs/lung_ct.cond/ckpt_300000.pth'

    for n_steps in [10, 50, 250]:
        project_root = f'/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT.cond/exp1.{n_steps}'
        os.makedirs(project_root, exist_ok=True)
        h5_dir = os.path.join(project_root, 'h5_internal_evaluation_v2')
        if os.path.exists(h5_dir):
            os.remove(h5_dir)
        ln_cmd = f'ln -sf {source_h5_dir} {h5_dir}'
        os.system(ln_cmd)
        out_h5_dir = os.path.join(project_root, 'h5_internal_evaluation.fov_extended')
        preview_dir = os.path.join(project_root, 'result_preview_png')

        with open(yml_config, "r") as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)

        runner_utils = InternalEvaluationUtilsRePaint(config, ckpt_path, h5_dir, out_h5_dir)
        runner_utils.run_inference(
            preview_dir=preview_dir,
            n_steps=n_steps
        )


if __name__ == "__main__":
    run_inpainting()


