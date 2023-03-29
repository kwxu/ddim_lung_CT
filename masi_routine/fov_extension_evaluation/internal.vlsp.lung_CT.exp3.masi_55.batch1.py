import h5py
import os
import pandas as pd
from masi_routine.fov_extension_evaluation.eval_utils import InternalEvaluationUtils, dict2namespace, InternalEvaluationData
import yaml


def split_two_batch():
    project_root = '/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT/exp3'
    in_h5_dir = os.path.join(project_root, 'h5_internal_evaluation_v2')
    batch1_dir = os.path.join(project_root, 'h5_internal_evaluation_v2.batch1')
    os.makedirs(batch1_dir, exist_ok=True)
    batch2_dir = os.path.join(project_root, 'h5_internal_evaluation_v2.batch2')
    os.makedirs(batch2_dir, exist_ok=True)

    h5_file_list = os.listdir(in_h5_dir)
    for file_idx in range(0, len(h5_file_list) // 2):
        h5_filename = h5_file_list[file_idx]
        in_h5 = os.path.join(in_h5_dir, h5_filename)
        out_h5 = os.path.join(batch1_dir, h5_filename)
        ln_cmd = f'ln -sf {in_h5} {out_h5}'
        os.system(ln_cmd)

    for file_idx in range(len(h5_file_list) // 2, len(h5_file_list)):
        h5_filename = h5_file_list[file_idx]
        in_h5 = os.path.join(in_h5_dir, h5_filename)
        out_h5 = os.path.join(batch2_dir, h5_filename)
        ln_cmd = f'ln -sf {in_h5} {out_h5}'
        os.system(ln_cmd)


def run_repaint():
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs/lung_ct_full_body.exp3.narrow_range.yml'
    project_root = '/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT/exp3'
    ckpt_path = os.path.join(project_root, 'models/ckpt_265000.pth')
    h5_dir = os.path.join(project_root, 'h5_internal_evaluation_v2.batch1')
    preview_dir = os.path.join(project_root, 'result_preview_png')

    with open(yml_config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    runner_utils = InternalEvaluationUtils(config, ckpt_path, h5_dir)
    runner_utils.run_inference(
        preview_dir=preview_dir
    )


def plot_result_single():
    hdf5_filename = '00000159time20151006_T10.hdf5'
    project_root = '/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT/exp1'
    h5_dir = os.path.join(project_root, 'h5_internal_evaluation_v2.batch1')
    out_dir = os.path.join(project_root, 'result_from_hdf5')

    data_obj = InternalEvaluationData(h5_dir, corrupt_val=-1)
    data_obj.plot_inference_result_from_hdf5(hdf5_filename, out_dir)


if __name__ == "__main__":
    # split_two_batch()
    run_repaint()
    # plot_result_single()

