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
    yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs/lung_ct_full_body.yml'

    # parse config file
    with open(yml_config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    lung_config = new_config.lung_ct_config
    test_h5_dir = os.path.join(lung_config.data_dir, 'h5_internal_evaluation_v2')
    h5_list = os.listdir(test_h5_dir)

    withheld_test_utils = WithheldTestUtils(lung_config)
    out_png_dir = "/local_storage/xuk9/Projects/DDIM_lung_CT/lung_CT/check_test_cases/raw"
    os.makedirs(out_png_dir, exist_ok=True)
    for h5_file_name in tqdm(h5_list, total=len(h5_list)):
        withheld_test_utils.check_single_raw(h5_file_name, os.path.join(out_png_dir, h5_file_name))


if __name__ == "__main__":
    sys.exit(main())