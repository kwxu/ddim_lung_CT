import sys
import os

import numpy as np
import yaml
import argparse
from tqdm import tqdm
from datasets.utils_lung import WithheldTestUtils
from runners.diffusion import SampleSpecificUtils


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def outpainting_square():
    for ratio_idx, ratio in zip([0, 1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6, 0.5]):
        # Get the x0_gt and mask
        withheld_test_utils = WithheldTestUtils(lung_config)
        x0_gt, mask_gt = withheld_test_utils.get_corrupt_sample_w_square_mask(
            test_case_h5,
            square_ratio=ratio
        )
        x0_gt = x0_gt[np.newaxis, np.newaxis, :, :]
        mask_gt = mask_gt[np.newaxis, np.newaxis, :, :]

        # Run prediction
        exp_path = "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/exp1"
        sample_utils = SampleSpecificUtils(
            config=config,
            exp_path=exp_path,
            log_name="lung_ct_full_body"
        )

        sample_utils.get_inpainting_x0_prediction(
            x0_gt,
            mask_gt,
            n_steps=50,
            n_resample=20,
            ckpt_id=ckpt_id,
            n_output_steps=10,
            output_dir=os.path.join(exp_path, 'inpainting', f'{test_case_h5}', f'ratio_{ratio_idx}')
        )


def inpainting_circle():
    for ratio_idx, ratio in zip([0, 1, 2, 3, 4, 5], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        # Get the x0_gt and mask
        withheld_test_utils = WithheldTestUtils(lung_config)
        x0_gt, mask_gt = withheld_test_utils.get_corrupt_sample_w_cen_circle(
            test_case_h5,
            circle_ratio=ratio
        )
        x0_gt = x0_gt[np.newaxis, np.newaxis, :, :]
        mask_gt = mask_gt[np.newaxis, np.newaxis, :, :]

        # Run prediction
        exp_path = "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/exp1"
        sample_utils = SampleSpecificUtils(
            config=config,
            exp_path=exp_path,
            log_name="lung_ct_full_body"
        )

        sample_utils.get_inpainting_x0_prediction(
            x0_gt,
            mask_gt,
            n_steps=50,
            n_resample=20,
            ckpt_id=ckpt_id,
            n_output_steps=10,
            output_dir=os.path.join(exp_path, 'inpainting.circle', f'{test_case_h5}', f'ratio_{ratio_idx}')
        )



if __name__ == "__main__":
    test_case_h5 = "00000003time20140109_T8.hdf5"
    ckpt_id = 360000
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs/lung_ct_full_body.yml'

    # parse config file
    with open(yml_config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    lung_config = config.lung_ct_config

    # outpainting_square()
    inpainting_circle()