import glob
import numpy as np
import pandas as pd
import os
import logging
import torch.utils.data as data
import random
from .utils_lung import get_dataset_utils, ImageTransformer, SyntheticMaskGenerator
import h5py



logger = logging.getLogger()


class ChestDataLoader(data.Dataset):
    def __init__(self, config):
        """
        config dict should be the one retrieved by tag "lung_ct_config" from the overall configuration structure.
        """
        self.config = config

        slice_record_df = pd.read_csv(
            os.path.join(config.data_dir, 'slice_record.csv')
        )

        # Here we use a combination of "train" and "valid"
        self.mode_slice_df = slice_record_df.loc[(slice_record_df["train"] == 1) | (slice_record_df["valid"] == 1)]

        self.mask_generator = SyntheticMaskGenerator(self.config)
        self.sample_generator = ImageTransformer(self.config)

        self.h5_dir = os.path.join(config.data_dir, 'h5_data')
        self.ds_utils = get_dataset_utils(config.dataset)

    def __len__(self):
        return len(self.mode_slice_df.index)

    def __getitem__(self, index):
        item_row = self.mode_slice_df.iloc[index]
        case_name = self.ds_utils.get_case_str_from_pid_date(item_row['pid'], item_row['date'])
        h5_path = os.path.join(self.h5_dir, f'{case_name}.hdf5')

        db = h5py.File(h5_path, 'r')

        idx_slice = int(item_row['idx_slice'])
        slice_img_dict = {}
        for img_flag in ['ct', 'body_mask']:
            slice_img_dict[img_flag] = db[img_flag][:, :, idx_slice]

        db.close()

        generated_sample_img, _, _ = self.sample_generator.generate_sample(slice_img_dict)
        generated_mask_img = self.mask_generator.generate_mask()

        generated_sample_img = np.float32(generated_sample_img)

        return generated_sample_img, generated_mask_img


