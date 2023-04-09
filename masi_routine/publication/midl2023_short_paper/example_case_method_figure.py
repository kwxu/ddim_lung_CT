import os
from tqdm import tqdm
import pandas as pd
from utils import ScanWrapper, load_json, clip_image
import numpy as np
from scipy.interpolate import interp1d
from cv2 import resize
import cv2
import h5py
from runners.inpainting import InpaintingSampleUtils
import yaml
import argparse
import torchvision.utils as tvu
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)



def get_pid_date_from_case_str(case_str):
    return int(case_str[:8]), int(case_str[12:20])


def get_round_mask(shape, offset_x, offset_y, r):
    center_x = int(round(shape[0] / 2)) + offset_x
    center_y = int(round(shape[1] / 2)) + offset_y
    xv, yv = np.meshgrid(range(shape[0]), range(shape[1]))
    dist_map = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)

    round_mask = np.zeros(shape, dtype=int)
    # print(dist_map.shape, round_mask.shape)
    round_mask[dist_map < r] = 1
    return round_mask


def process_single_slice(config, ct_slice, body_slice):
    """
    We assume in these cases the body is complete
    """
    clip_range = config['data']['clip_range']
    scale_range = config['data']['scale_range']
    resample_size = config['data']['image_size']

    raw_slice = np.clip(ct_slice, clip_range[0], clip_range[1])
    normalizer = interp1d(clip_range, scale_range)
    raw_slice = normalizer(raw_slice)
    raw_slice[body_slice == 0] = scale_range[0]

    # inter_order = config['data']['inter_order']
    out_slice = resize(raw_slice, (resample_size, resample_size), interpolation=cv2.INTER_LINEAR)
    out_slice = np.clip(out_slice, clip_range[0], clip_range[1])
    out_body = resize(body_slice, (resample_size, resample_size), interpolation=cv2.INTER_NEAREST)

    return out_slice, out_body


def generate_demo_sample_h5():
    demo_sample_h5_dir = os.path.join(project_dir, 'h5_dir')
    os.makedirs(demo_sample_h5_dir, exist_ok=True)

    for demo_record in tqdm(demo_record_df.iterrows(), total=len(demo_record_df.index)):
        demo_record_dict = demo_record[1]
        case_name = demo_record_dict['case_name']
        crop_type = demo_record_dict['type']
        level = demo_record_dict['level']
        crop_ratio = demo_record_dict['ratio']

        nii_file_name = f'{case_name}.nii.gz'
        ct_img = ScanWrapper(os.path.join(raw_ct_dir, nii_file_name)).get_data()
        body_img = ScanWrapper(os.path.join(raw_body_dir, nii_file_name)).get_data()

        json_file_name = f'{case_name}_pred.json'
        vertloc_pred_dict = load_json(os.path.join(vertloc_pred_dir, json_file_name))
        level_idx = int(round(vertloc_pred_dict[level]['unit']))

        raw_ct_slice = clip_image(ct_img, 'axial', level_idx)
        raw_body_slice = clip_image(body_img, 'axial', level_idx)

        # Need to first process the raw ct image.
        ct_slice, body_slice = process_single_slice(args, raw_ct_slice, raw_body_slice)

        # Then, generate the FOV mask based on input configuration.
        clip_size = int(round(crop_ratio * ct_slice.shape[0]))
        if crop_type == 'square':
            fov_mask = np.zeros(ct_slice.shape, dtype=int)
            conner_loc = int(round(fov_mask.shape[0] - clip_size) / 2.)
            fov_mask[conner_loc:conner_loc+clip_size, conner_loc:conner_loc+clip_size] = 1
        else:
            fov_mask = get_round_mask(ct_slice.shape, 0, 0, int(round(clip_size / 2.)))

        h5_path = os.path.join(demo_sample_h5_dir, f'{case_name}.hdf5')
        # Here we only assign those used in the combined internal inference pipeline
        db = h5py.File(h5_path, 'a')
        db.attrs['n_slice'] = 1
        if 'sample' not in db:
            sample_grp = db.create_group('sample')
        else:
            sample_grp = db['sample']

        if 'n_sample' in sample_grp.attrs:
            sample_grp.attrs['n_sample'] += 1
        else:
            sample_grp.attrs['n_sample'] = 1

        sample_idx = sample_grp.attrs['n_sample'] - 1
        sample_name = f'sample{sample_idx}'
        single_sample_grp = sample_grp.create_group(sample_name)

        for ds_name, ds_data in zip(['ct', 'body_mask'], [ct_slice, body_slice]):
            ds_data = np.expand_dims(np.expand_dims(ds_data, axis=0), axis=0)
            single_sample_grp.create_dataset(ds_name, data=ds_data)

        single_sample_grp.create_dataset('fov_mask', data=np.expand_dims(fov_mask, axis=0))
        pid, date = get_pid_date_from_case_str(case_name)
        db.attrs['pid'] = pid
        db.attrs['date'] = date
        # db.attrs['slice_idx_list'] = [level_idx]
        # db.attrs['level'] = level
        single_sample_grp.attrs['slice_idx_list'] = [level_idx]
        single_sample_grp.attrs['level'] = level
        # if 'level' not in db.attrs:
        #     db.attrs['level'] = [level]
        # db.attrs['level'].append(level)
        db.close()


class SampleEvaluationUtils:
    def __init__(self):
        yml_config = '/nfs/masi/xuk9/src/ddim_lung_CT/configs/lung_ct_full_body.exp1.yml'
        project_root = '/nfs/masi/xuk9/Projects/ChestExtrapolation/ddim_lung_CT/exp1'
        ckpt_path = os.path.join(project_root, 'models/ckpt_360000.pth')

        with open(yml_config, "r") as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)

        self.sample_utils = InpaintingSampleUtils(self.config)
        self.sample_utils.load_model(ckpt_path)

    def generate_sample_trajectory(self, case_name, sample_name):
        h5_path = os.path.join(project_dir, 'h5_dir', f'{case_name}.hdf5')

        db = h5py.File(h5_path, 'r')
        sample_grp = db['sample'][sample_name]
        ct_slice = sample_grp['ct'][0, 0, :, :]
        fov_mask = sample_grp['fov_mask'][0, :, :]
        db.close()

        # print(ct_slice.shape)
        # print(fov_mask.shape)

        corrupt_slice = ct_slice.copy()
        corrupt_slice[fov_mask == 0] = 0
        ct_slice = ct_slice[None, None, :, :]
        ct_slice = data_transform(self.config, ct_slice)
        fov_mask = fov_mask[None, None, :, :]

        # n_steps = 50
        # n_resample = 20
        # forward_xts = self.sample_utils.generate_forward_steps(ct_slice, n_steps)
        #
        # xs, backward_xts = self.sample_utils.run_inference(
        #     x0_gt=ct_slice,
        #     mask_gt=fov_mask,
        #     # n_steps=50,
        #     # n_resample=20,
        #     n_steps=n_steps,
        #     n_resample=n_resample,
        #     last_only=False
        # )

        out_png_dir = os.path.join(project_dir, 'example_png', f'{case_name}_{sample_name}')
        os.makedirs(out_png_dir, exist_ok=True)
        print(f'Save to {out_png_dir}')

        def save_png(img, out_png):
            img = inverse_data_transform(self.config, img)
            tvu.save_image(img, out_png)

        def save_png_w_mask(img, mask, out_png, remove_background=False):
            # img = inverse_data_transform(self.config, img)
            # img = img.numpy()

            if remove_background:
                img[mask == 0] = 0

            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(
                img,
                interpolation='bilinear',
                cmap='gray',
                norm=colors.Normalize(vmin=0, vmax=1))

            mask = mask.astype(float)
            mask[mask == 0] = np.nan
            ax.imshow(
                mask,
                cmap='Oranges',
                norm=colors.Normalize(vmin=0, vmax=1),
                alpha=0.3
            )
            plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

        # save_png_w_mask(fov_mask[0, 0, :, :], fov_mask[0, 0, :, :], os.path.join(out_png_dir, 'mask.png'))
        save_png_w_mask(corrupt_slice, fov_mask[0, 0, :, :], os.path.join(out_png_dir, 'corrupt.png'))

        # save_png(torch.from_numpy(ct_slice[0, 0, :, :]), os.path.join(out_png_dir, 'gt.png'))
        # tvu.save_image(torch.from_numpy(corrupt_slice), os.path.join(out_png_dir, 'input.png'))
        #
        # for step_idx in range(n_steps):
        #     # save_png(forward_xts[step_idx], os.path.join(out_png_dir, f'forward_step{step_idx}.png'))
        #     # save_png(backward_xts[step_idx], os.path.join(out_png_dir, f'backward_step{step_idx}.png'))
        #     save_png_w_mask(
        #         forward_xts[step_idx][0, 0, :, :],
        #         fov_mask[0, 0, :, :],
        #         os.path.join(out_png_dir, f'forward_step{step_idx}.png'), remove_background=True)
        #     save_png_w_mask(
        #         backward_xts[step_idx][0, 0, :, :],
        #         fov_mask[0, 0, :, :],
        #         os.path.join(out_png_dir, f'backward_step{step_idx}.png'))
        #
        # save_png(xs[-1][0, 0, :, :], os.path.join(out_png_dir, 'pred.png'))
        # save_png(xs[0][0, 0, :, :], os.path.join(out_png_dir, 'forward_seed.png'))
        #
        # tvu.save_image(torch.from_numpy(fov_mask).float(), os.path.join(out_png_dir, 'mask.png'))


if __name__ == '__main__':
    project_dir = '/local_storage/xuk9/Projects/DDIM_lung_CT/Publication/midl2023/method'
    os.makedirs(project_dir, exist_ok=True)

    raw_ct_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/VLSP_all/ct_std'
    raw_body_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/VLSP_all/body_mask'
    vertloc_pred_dir = '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/VLSP_all/vertloc_pred'

    args = {
        'data': {
            'clip_range': [-1000, 600],
            'scale_range': [0, 1],
            'image_size': 256
        }
    }

    demo_record_csv = os.path.join(project_dir, 'demo_cases.csv')
    print(f'Load {demo_record_csv}')
    demo_record_df = pd.read_csv(demo_record_csv)

    # generate_demo_sample_h5()

    sample_utils = SampleEvaluationUtils()
    # sample_utils.generate_sample_trajectory('00000459time20180425', 'sample0')
    # sample_utils.generate_sample_trajectory('00000459time20180425', 'sample3')
    sample_utils.generate_sample_trajectory('00000489time20180223', 'sample0')
