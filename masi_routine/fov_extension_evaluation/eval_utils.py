import os
import torch
import torch.utils.data as data
import glob
import numpy as np
import h5py
from tqdm import tqdm
from runners.inpainting import InpaintingSampleUtils
import argparse
from scipy.interpolate import interp1d
import cv2


def save_img_stack_hdf5_grp(target_grp, img_stack, ds_name):
    if ds_name in target_grp:
        del target_grp[ds_name]

    chunk_shape = list(img_stack.shape)
    chunk_shape[0] = 1
    chunk_shape = tuple(chunk_shape)
    target_grp.create_dataset(
        ds_name,
        data=img_stack,
        chunks=chunk_shape,
        compression='gzip'
    )


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def plot_inference_result(corrupt_img, pred_img, out_png):
    combined_img = np.concatenate([corrupt_img, pred_img], axis=1)

    scale_range = [-1, 1]
    combined_img = np.clip(combined_img, scale_range[0], scale_range[1])
    normalizer = interp1d(scale_range, [0, 255])
    combined_img = normalizer(combined_img).astype(np.uint8)

    combined_img = np.repeat(combined_img[:, :, np.newaxis], 3, axis=2)

    cv2.imwrite(out_png, combined_img)


class InternalEvaluationData:
    def __init__(self, h5_dir, corrupt_val):
        self.h5_dir = h5_dir
        self.h5_list = glob.glob(f'{self.h5_dir}/*.hdf5')
        self.corrupt_val = corrupt_val

    def __len__(self):
        return len(self.h5_list)

    # def __getitem__(self, item):
    def get_item(self, item):
        h5_path = self.h5_list[item]

        ds_stack_dict = {
            'sample': [],
            'mask': [],
            'corrupt': []
        }

        db = h5py.File(h5_path, 'r')
        # n_slice = db.attrs['n_slice']
        n_sample = db['sample'].attrs['n_sample']
        for idx_sample in range(n_sample):
            sample_name = f'sample{idx_sample}'
            single_sample_grp = db['sample'][sample_name]
            # sample_ct_stack = single_sample_grp['ct'][:]
            sample_ct_stack = single_sample_grp['ct'][2, 0, :, :]
            sample_ct_stack = np.repeat(sample_ct_stack[np.newaxis, :, :], 1, axis=0)
            sample_ct_stack = np.repeat(sample_ct_stack[np.newaxis, :, :, :], 1, axis=0)
            fov_mask = single_sample_grp['fov_mask'][0, :, :]
            # sample_mask_stack = np.repeat(fov_mask[np.newaxis, :, :, :], n_slice, axis=0)
            sample_mask_stack = np.repeat(fov_mask[np.newaxis, :, :], 1, axis=0)
            sample_mask_stack = np.repeat(sample_mask_stack[np.newaxis, :, :, :], 1, axis=0)
            sample_corrupt_stack = sample_ct_stack.copy()
            sample_corrupt_stack[sample_mask_stack == 0] = self.corrupt_val

            ds_stack_dict['sample'].append(sample_ct_stack)
            ds_stack_dict['mask'].append(sample_mask_stack)
            ds_stack_dict['corrupt'].append(sample_corrupt_stack)
        db.close()

        # for ds_name in ds_stack_dict.keys():
        #     ds_stack_dict[ds_name] = np.concatenate(ds_stack_dict[ds_name], axis=0)

        data_dict = ds_stack_dict
        data_dict.update({
            'h5_filename': os.path.basename(h5_path)
        })

        return data_dict

    def save_inference_result(self, img_stack_dict, h5_filename):
        h5_path = os.path.join(self.h5_dir, h5_filename)

        db = h5py.File(h5_path, 'a')
        n_slice = db.attrs['n_slice']
        n_sample = db['sample'].attrs['n_sample']

        for idx_sample in range(n_sample):
            sample_name = f'sample{idx_sample}'
            single_sample_grp = db['sample'][sample_name]

            idx_start = n_slice * idx_sample
            idx_end = n_slice * (idx_sample + 1)

            predict_stack = img_stack_dict['predict'][idx_start:idx_end, :, :, :]
            save_img_stack_hdf5_grp(single_sample_grp, predict_stack, 'predict')

        db.close()

    def save_inference_mid_slice_only(self, pred_list, h5_filename):
        h5_path = os.path.join(self.h5_dir, h5_filename)

        db = h5py.File(h5_path, 'a')
        # n_slice = db.attrs['n_slice']
        n_sample = db['sample'].attrs['n_sample']

        for idx_sample in range(n_sample):
            sample_name = f'sample{idx_sample}'
            single_sample_grp = db['sample'][sample_name]

            # pred_img = pred_list[idx_sample, :, :, :]
            pred_img = pred_list[idx_sample]
            save_img_stack_hdf5_grp(single_sample_grp, pred_img, 'predict')

        db.close()

    def plot_inference_result_from_hdf5(self, hdf5_filename, out_dir):
        case_name = hdf5_filename.replace('.hdf5', '')
        os.makedirs(out_dir, exist_ok=True)
        print(f'Save to {out_dir}')
        h5_path = os.path.join(self.h5_dir, hdf5_filename)

        db = h5py.File(h5_path, 'r')
        n_sample = db['sample'].attrs['n_sample']
        for idx_sample in range(n_sample):
            sample_name = f'sample{idx_sample}'
            single_sample_grp = db['sample'][sample_name]
            sample_ct_img = single_sample_grp['ct'][2, 0, :, :]
            fov_mask = single_sample_grp['fov_mask'][0, :, :]
            corrupt_img = sample_ct_img.copy()
            corrupt_img[fov_mask == 0] = -1
            pred_ct_img = single_sample_grp['predict'][0, 0, :, :]
            out_png = os.path.join(out_dir, f'{case_name}_{idx_sample}.png')
            plot_inference_result(
                corrupt_img, pred_ct_img,
                out_png
            )
        db.close()


class InternalEvaluationUtilsRePaint:
    def __init__(self, config, ckpt_path, hdf5_data_dir):
        self.config = config
        self.dataset = InternalEvaluationData(hdf5_data_dir, corrupt_val=-1)
        self.sample_util = InpaintingSampleUtils(self.config)
        self.sample_util.load_model(ckpt_path)

        # self.sample_n_steps = n_steps
        # self.sample_n_resample = 20

    def run_inference(self, preview_dir=None, n_steps=50):
        # How many sample we have
        # n_sample = 0
        # for item_idx in tqdm(range(len(self.dataset)), total=len(self.dataset)):
        #     img_data = self.dataset.get_item(item_idx)
        #     n_sample += len(img_data['sample'])
        #
        # print(f'Number of samples: {n_sample}')

        if preview_dir is not None:
            print(f'Save result images to {preview_dir}')
            os.makedirs(preview_dir, exist_ok=True)

        for item_idx in range(len(self.dataset)):
            img_data = self.dataset.get_item(item_idx)
            n_sample = len(img_data['sample'])

            pred_list = []
            for sample_idx in tqdm(range(n_sample), total=n_sample,
                                   desc=f'Process item {item_idx} ({len(self.dataset)})'):
                x0_pred = self.sample_util.run_inference(
                    x0_gt=img_data['corrupt'][sample_idx],
                    mask_gt=img_data['mask'][sample_idx],
                    n_steps=n_steps
                    # n_resample=self.sample_n_resample
                )

                pred_list.append(x0_pred)
                # pred_list.append(img_data['corrupt'][sample_idx])

                if preview_dir is not None:
                    case_name = img_data['h5_filename'].replace('.hdf5', '')
                    out_png = os.path.join(preview_dir, f'{case_name}_{sample_idx}.png')
                    plot_inference_result(
                        img_data['corrupt'][sample_idx][0, 0, :, :],
                        # pred_list[sample_idx][0, 0, :, :],
                        x0_pred[0, 0, :, :],
                        out_png
                    )

            h5_filename = img_data['h5_filename']
            self.dataset.save_inference_mid_slice_only(
                pred_list,
                h5_filename
            )
