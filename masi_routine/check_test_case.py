from datasets.utils_lung import WithheldTestUtils
import sys
import os
import yaml
import argparse
from tqdm import tqdm


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
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs/lung_ct_full_body.exp1.yml'

    # parse config file
    with open(yml_config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    lung_config = new_config.lung_ct_config
    test_h5_dir = os.path.join(lung_config.data_dir, 'h5_internal_evaluation_v2')
    h5_list = os.listdir(test_h5_dir)

    withheld_test_utils = WithheldTestUtils(lung_config)

    # out_png_dir = "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/check_test_cases/raw"
    # os.makedirs(out_png_dir, exist_ok=True)
    # for h5_file_name in tqdm(h5_list, total=len(h5_list)):
    #     withheld_test_utils.check_single_raw(h5_file_name, os.path.join(out_png_dir, h5_file_name))

    # withheld_test_utils.show_out_h5_hierarchy("00000003time20140109_T8")
    # out_png_dir = "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/check_test_cases/sample"
    # for h5_case_name in ["00000003time20140109_T8"]:
    #     for sample_idx in range(10):
    #         sample_name = f'sample{sample_idx}'
    #         sample_output_dir = os.path.join(out_png_dir, f'{h5_case_name}', sample_name)
    #         os.makedirs(sample_output_dir, exist_ok=True)
    #
    #         withheld_test_utils.check_single_sample(h5_case_name, sample_name, sample_output_dir)

    # for ratio_idx, ratio in zip([0, 1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6, 0.5]):
    #     withheld_test_utils.get_corrupt_sample_w_square_mask(
    #         h5_file_name='00000003time20140109_T8.hdf5',
    #         square_ratio=ratio,
    #         output_dir=os.path.join(
    #             "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/check_test_cases/sample",
    #             "square_corruption",
    #             f'ratio_{ratio_idx}'
    #         )
    #     )

    for ratio_idx, ratio in zip([0, 1, 2, 3, 4, 5], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        withheld_test_utils.get_corrupt_sample_w_cen_circle(
            h5_file_name='00000003time20140109_T8.hdf5',
            circle_ratio=ratio,
            output_dir=os.path.join(
                "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/check_test_cases/sample/circle_corruption",
                f'ratio_{ratio_idx}'
            )
        )


if __name__ == "__main__":
    sys.exit(main())