import sys
import os
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


def main():
    test_case_h5 = "00000003time20140109_T8.hdf5"
    ckpt_list = [360000]
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs/lung_ct_full_body.yml'

    # parse config file
    with open(yml_config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    lung_config = config.lung_ct_config

    # Get the x0
    withheld_test_utils = WithheldTestUtils(lung_config)
    x0 = withheld_test_utils.get_single_raw_input_format(test_case_h5)

    # Run prediction
    exp_path = "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/exp1"
    sample_utils = SampleSpecificUtils(
        config=config,
        exp_path=exp_path,
        log_name="lung_ct_full_body"
    )
    show_ts = [50, 100, 200, 400, 600, 800, 900, 950, 1000]
    # show_ts = [1, 2]
    show_ts = [t - 1 for t in show_ts]
    # sample_utils.get_x0_prediction(
    #     x0,
    #     run_ts=show_ts,
    #     show_ts=show_ts,
    #     show_ckpts=ckpt_list,
    #     out_png_dir=os.path.join(exp_path, 'image_samples', 'x0_prediction_test_case')
    # )

    sample_utils.get_x0_prediction_along_trajectory(
        x0,
        show_ts,
        ckpt_list[0],
        os.path.join(exp_path, 'image_samples', 'x0_prediction_trajectory')
    )



if __name__ == "__main__":
    sys.exit(main())