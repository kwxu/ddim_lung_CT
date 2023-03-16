import glob
import numpy as np
import pandas as pd
import os
import logging
import torch.utils.data as data
import random
from .utils_lung import get_dataset_utils
import h5py
from scipy.interpolate import interp1d
from .utils_lung import get_body_mask_bb
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate, warp, SimilarityTransform
from skimage.transform import resize
from .utils_lung import get_random_apply_flag
from .utils_lung import get_symmetric_pad_larger_dim
from .utils_lung import save_img_stack_hdf5_grp


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
        generated_sample_img = np.float32(generated_sample_img)

        return generated_sample_img, 0


class ImageTransformer:
    def __init__(self, config):
        self.config = config

    def generate_random_config(self):
        angle_max = self.config.augmentation.rotation_degree
        pad_ratio_range = self.config.augmentation.padding_ratio
        trans_max_ratio_x = self.config.augmentation.trans_ratio_x
        trans_max_ratio_y = self.config.augmentation.trans_ratio_y

        return {
                'gaussian_blurry': get_random_apply_flag(self.config.augmentation.gaussian_smooth_p),
                'random_rotation': random.uniform(-angle_max, angle_max),
                'pad_ratio': random.uniform(pad_ratio_range[0], pad_ratio_range[1]),
                'trans_ratio_x': random.uniform(-trans_max_ratio_x, trans_max_ratio_x),
                'trans_ratio_y': random.uniform(-trans_max_ratio_y, trans_max_ratio_y)}

    def generate_sample(self, slice_img_dict, random_config=None):
        if random_config is None:
            random_config = self.generate_random_config()

        ct_image = np.rot90(slice_img_dict['ct'])
        body_image = np.rot90(slice_img_dict['body_mask'])

        # Crop using the body mask bounding box.
        ct_image, body_image = self._crop_body_bbox(ct_image, body_image)

        # Random apply the gaussian blurry
        if random_config['gaussian_blurry']:
            ct_image = self._apply_gaussian_blurry(ct_image)

        # Normalize the intensity scale.
        ct_image = self._run_preprocessing(ct_image, body_image)
        pad_val = self.config.scale_range[0]

        # Random rotation
        ct_image, body_image = self._apply_random_rot_angle(
            ct_image, body_image, pad_val, random_config['random_rotation'])

        # Re-cropping with body bbox
        ct_image, body_image = self._crop_body_bbox(ct_image, body_image)

        # Square padding, determined by random scaling
        ct_image = self._apply_square_padding(ct_image, pad_val, random_config['pad_ratio'])

        # Random translation
        ct_image = self._apply_xy_translation(
            ct_image, pad_val, random_config['trans_ratio_x'], random_config['trans_ratio_y'])

        # Resize the image, log down the scale-ratio here.
        ori_dim = ct_image.shape[0]
        resize_dim = self.config.image_size
        inter_order = self.config.inter_order
        ct_image = resize(ct_image, (resize_dim, resize_dim), order=inter_order)
        scale_ratio = ori_dim / resize_dim

        # Reshape the image for torch (C, H, W)
        ct_image = np.repeat(ct_image[np.newaxis, :, :], self.config.channels, axis=0)

        return ct_image, random_config, scale_ratio

    def apply_random_trans_stack(self, slice_stack, body_stack, random_config, use_body_slice_idx=2):
        slice_stack = slice_stack.copy()
        body_stack = body_stack.copy()
        trans_list = []
        scale_ratio = None
        for idx_slice in range(slice_stack.shape[0]):
            trans_slice, _, scale_ratio = self.generate_sample(
                {
                    'ct': slice_stack[idx_slice],
                    # 'body_mask': body_stack[idx_slice]
                    'body_mask': body_stack[use_body_slice_idx]
                }, random_config)
            trans_list.append(trans_slice)

        return np.stack(trans_list, axis=0), scale_ratio

    def apply_random_trans_to_mask_stack(self, mask_stack, body_stack, random_config, use_body_slice_idx=2):
        mask_stack = mask_stack.copy()
        body_stack = body_stack.copy()
        trans_list = []
        for idx_slice in range(mask_stack.shape[0]):
            trans_slice = self.apply_random_trans_to_mask(
                {
                    'in_mask': mask_stack[idx_slice],
                    # 'body_mask': body_stack[idx_slice]
                    'body_mask': body_stack[use_body_slice_idx]
                }, random_config)
            trans_list.append(trans_slice)

        return np.stack(trans_list, axis=0)

    def apply_random_trans_to_mask(self, slice_img_dict, random_config):
        in_mask_image = np.rot90(slice_img_dict['in_mask'])
        body_image = np.rot90(slice_img_dict['body_mask'])

        in_mask_image, body_image = self._crop_body_bbox(in_mask_image, body_image)
        in_mask_image, body_image = self._apply_random_rot_angle(
            in_mask_image, body_image, 0, random_config['random_rotation'])

        in_mask_image, body_image = self._crop_body_bbox(in_mask_image, body_image)
        in_mask_image = self._apply_square_padding(in_mask_image, 0, random_config['pad_ratio'])

        in_mask_image = self._apply_xy_translation(
            in_mask_image, 0, random_config['trans_ratio_x'], random_config['trans_ratio_y'])

        resize_dim = self.config.image_size
        in_mask_image = resize(in_mask_image, (resize_dim, resize_dim), order=0)
        in_mask_image = np.repeat(in_mask_image[np.newaxis, :, :], 3, axis=0)

        return in_mask_image

    def _apply_gaussian_blurry(self, ct_image):
        return gaussian_filter(ct_image, sigma=self.config.augmentation.gaussian_smooth_sigma)

    def _apply_random_rot_angle(self, ct_image, body_image, pad_val, angle_val):
        inter_order = self.config.inter_order
        ct_rot = rotate(ct_image, angle_val, resize=True, mode='constant', cval=pad_val, preserve_range=True, order=inter_order)
        mask_rot = rotate(body_image.astype(bool), angle_val, resize=True, mode='constant', cval=0, order=0).astype(int)

        return ct_rot, mask_rot

    @staticmethod
    def _crop_body_bbox(ct_image, body_image):
        body_bbox = get_body_mask_bb(body_image)
        ct_image = ct_image[body_bbox[0]:body_bbox[2], body_bbox[1]:body_bbox[3]]
        body_image = body_image[body_bbox[0]:body_bbox[2], body_bbox[1]:body_bbox[3]]
        return ct_image, body_image

    def _run_preprocessing(self, ct_image, body_image):
        clip_range = self.config.clip_range
        scale_range = self.config.scale_range
        ct_image[body_image == 0] = clip_range[0]
        ct_image = np.clip(ct_image, clip_range[0], clip_range[1])
        normalizer = interp1d(clip_range, scale_range)
        ct_image = normalizer(ct_image)

        return ct_image

    @staticmethod
    def _apply_square_padding(ct_image, pad_val, pad_ratio):
        # Apply pad, pad_val should be the air intensity value (scaled / non-scaled)
        padded_image = get_symmetric_pad_larger_dim(ct_image, pad_ratio, pad_val)

        return padded_image

    def _apply_xy_translation(self, ct_image, pad_val, trans_ratio_x, trans_ratio_y):
        img_dim = ct_image.shape[0]
        trans_dim_x = int(round(img_dim * trans_ratio_x))
        trans_dim_y = int(round(img_dim * trans_ratio_y))

        tform = SimilarityTransform(translation=(trans_dim_x, trans_dim_y))
        inter_order = self.config.inter_order
        trans_image = warp(ct_image, tform, mode='constant', cval=pad_val, order=inter_order)

        return trans_image
