import os
import sys
import json
import random
import logging
from typing import Dict, Union, Optional
from types import SimpleNamespace
import torch
import numpy as np
import json
import yaml
import pandas as pd
import glob
import pprint
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from skimage.measure import regionprops
import random
import cv2
from scipy.interpolate import interp1d
import logging
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib import colors


logger = logging.getLogger()


def load_yaml_config(yaml_config):
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_save_path(config):
    save_dir = os.path.join(config['env']['project_dir'], 'train_output')
    os.makedirs(save_dir, exist_ok=True)
    num_existing_dirs = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, "run_{}".format(num_existing_dirs))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_args(config, save_path: str):
    args_file_path = os.path.join(save_path, "args.json")
    with open(args_file_path, "w") as file:
        json.dump(config, file, indent=4)


def save_state(
        epoch: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        path: str,
        filename: str = "best_model.tar",
):
    old_checkpoint_files = list(
        filter(lambda x: "checkpoint" in x, os.listdir(path))
    )

    state_dict = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
    }
    file_path = os.path.join(path, filename)
    logger.info("Save current state to {}".format(filename))
    torch.save(state_dict, file_path)

    for file in old_checkpoint_files:
        os.remove(os.path.join(path, file))


def load_dataset_indices(load_path: str, file_name: str = "indices.json"):
    with open(os.path.join(load_path, file_name), "r") as file:
        indices = json.load(file)
    return indices


def load_state(path: str, map_location=None):
    loaded_state = torch.load(path, map_location=map_location)
    logger.info(
        "Loaded state from {} saved at epoch {}".format(path, loaded_state["epoch"])
    )
    return loaded_state


def load_args(run_path):
    run_args = json.load(open(os.path.join(run_path, "args.json")))
    return SimpleNamespace(**run_args)


class AverageMeter:
    """
    AverageMeter implements a class which can be used to track a metric over the entire training process.
    (see https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all class variables to default values
        """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates class variables with new value and weight
        """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        """
        Implements format method for printing of current AverageMeter state
        """
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )


class AverageMeterSet:
    """
    AverageMeterSet implements a class which can be used to track a set of metrics over the entire training process
    based on AverageMeters (Source: https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


def set_device(config):
    if torch.cuda.is_available() and (config['env']['device'] == 'cuda'):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def initialize_logger(save_path: str, log_file: str):
    logger = logging.getLogger()
    logging.basicConfig(
        filename=os.path.join(save_path, log_file),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


class DatasetUtils:
    @staticmethod
    def get_pid_date_from_case_str(case_str):
        raise NotImplementedError

    @staticmethod
    def get_case_str_from_pid_date(pid, date):
        raise NotImplementedError


class VLSPUtils(DatasetUtils):
    @staticmethod
    def get_pid_date_from_case_str(case_str):
        return int(case_str[:8]), int(case_str[12:20])

    @staticmethod
    def get_case_str_from_pid_date(pid, date):
        return f'{pid:08d}time{date}'


class NLSTUtils(DatasetUtils):
    @staticmethod
    def get_pid_date_from_case_str(case_str):
        return int(case_str[:6]), int(case_str[10:14])

    @staticmethod
    def get_case_str_from_pid_date(pid, date):
        return f'{pid}time{date}'


def get_dataset_utils(dataset_name):
    if dataset_name == 'vlsp':
        return VLSPUtils()
    elif dataset_name == 'nlst':
        return NLSTUtils()
    else:
        raise NotImplementedError


# From: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


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


def to_items(dic):
    return dict(map(_to_item, dic.items()))


def _to_item(item):
    return item[0], item[1].item()



def normalize_image_intensity(img, clip_range, scale_range):
    img = np.clip(img, clip_range[0], clip_range[1])
    normalizer = interp1d(clip_range, scale_range)
    return normalizer(img)


def get_body_mask_bb(body_mask):
    progs = regionprops(body_mask.astype(int))
    return progs[0].bbox


def get_random_apply_flag(p_value):
    random_val = random.uniform(0, 1)
    return random_val < p_value


def get_symmetric_pad_larger_dim(in_image, pad_ratio, pad_val):
    body_dim = np.max(in_image.shape)
    pad_dim_total = int(round(pad_ratio * body_dim))
    train_sample_image = np.zeros((pad_dim_total, pad_dim_total), dtype=float)
    train_sample_image.fill(pad_val)
    x_low = int(round((pad_dim_total - in_image.shape[0]) / 2))
    y_low = int(round((pad_dim_total - in_image.shape[1]) / 2))
    train_sample_image[x_low:(x_low + in_image.shape[0]), y_low:(y_low + in_image.shape[1])] = in_image[:, :]

    return train_sample_image


def get_round_mask(shape, offset_x, offset_y, r):
    center_x = int(round(shape[0] / 2)) + offset_x
    center_y = int(round(shape[1] / 2)) + offset_y
    xv, yv = np.meshgrid(range(shape[0]), range(shape[1]))
    dist_map = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)

    round_mask = np.zeros(shape, dtype=int)
    # print(dist_map.shape, round_mask.shape)
    round_mask[dist_map < r] = 1
    return round_mask


def save_png_w_2d_npy(img, clip_range, out_png):
    img = np.clip(img, clip_range[0], clip_range[1])
    normalizer = interp1d(clip_range, [0, 255])
    img = normalizer(img).astype(np.uint8)
    # print(f'Save to {out_png}')
    cv2.imwrite(out_png, img)


def get_tci_value_stack(body_stack, fov_mask):
    n_slice = body_stack.shape[0]
    tci_val_list = []
    fov_mask = fov_mask.copy()[0, :, :]
    for idx_slice in range(n_slice):
        body_mask = body_stack[idx_slice, 0, :, :]
        corrupt_body_img = body_mask.copy().astype(np.int)
        ppr_mask = 1 - fov_mask
        corrupt_body_img[ppr_mask == 1] = 0
        tci_val = get_tci_value(corrupt_body_img, ppr_mask)
        tci_val_list.append(tci_val)

    return tci_val_list


def get_tci_value(body_mask, ppr_mask, boundary_width=2):
    body_boundary = body_mask.copy().astype(int)
    body_boundary[binary_erosion(body_boundary, iterations=boundary_width)] = 0
    fov_boundary = 1 - ppr_mask.copy().astype(int)
    fov_boundary[binary_erosion(fov_boundary, iterations=boundary_width)] = 0

    fake_body_boundary_value = 2
    body_boundary[(body_boundary == 1) & (fov_boundary == 1)] = fake_body_boundary_value

    num_total_segment = np.count_nonzero(body_boundary)
    if num_total_segment == 0:
        return 0

    num_fake_segment = np.count_nonzero(body_boundary == fake_body_boundary_value)
    tci_val = num_fake_segment / num_total_segment

    return tci_val


def check_mask_boundary_extrude(in_mask):
    dilated_mask = binary_dilation(in_mask.astype(int), iterations=2)
    fake_ppr_mask = np.zeros(dilated_mask.shape, dtype=int)
    fake_tci_val = get_tci_value(dilated_mask, fake_ppr_mask)
    return fake_tci_val > 0


def get_ssim_score_list_stack(stack1, stack2):
    n_slice = stack1.shape[0]
    ssim_score_list = []
    for idx_slice in range(n_slice):
        ssim_score_list.append(
            structural_similarity(
                stack1[idx_slice, 0, :, :],
                stack2[idx_slice, 0, :, :]))

    return ssim_score_list


def get_dice(img1, img2):
    assert img1.shape == img2.shape

    img1 = img1.flatten().astype(float)
    img2 = img2.flatten().astype(float)

    dice_val = 2 * (img1 * img2).sum() / (img1 + img2).sum()

    return dice_val


def get_dsc_score_bc_slice(gt_slice, pred_slice):
    label_id_map = {
        'SAT': 1,
        'Muscle': 2}

    dsc_dict = {}
    for label_flag, label_id in label_id_map.items():
        bc_pred = (pred_slice == label_id).astype(int)
        bc_gt = (gt_slice == label_id).astype(int)
        dsc_dict[label_flag] = get_dice(bc_pred, bc_gt)

    mean_dsc = np.mean(list(dsc_dict.values()))
    return mean_dsc


def get_dsc_score_list_stack(gt_stack, pred_stack):
    """
    Here in, we only consider Muscle and Adipose tissue.
    :param gt_stack:
    :param pred_stack:
    :return:
    """
    n_slice = gt_stack.shape[0]
    dsc_list = []
    for idx_slice in range(n_slice):
        dsc_list.append(
            get_dsc_score_bc_slice(
                gt_stack[idx_slice, :, :],
                pred_stack[idx_slice, :, :]))

    return dsc_list


def get_body_mask(in_ct_slice_img, clip_range):
    """
    Assumptions of the input image:
    :param in_ct_slice_img:
    :param clip_range: e.g. [-1, 1], [-1000, 600]
    :return:
    """
    slice_img = np.clip(in_ct_slice_img, clip_range[0], clip_range[1])
    slice_img = interp1d(clip_range, [0, 1])(slice_img)

    thres_eps = 5e-2
    body_mask = np.zeros(slice_img.shape)
    body_mask[slice_img > thres_eps] = 1

    body_mask = binary_fill_holes(body_mask)

    return body_mask


def save_png_body_mask_overlay(ct_slice, ct_range, body_mask, out_png):
    ct_slice = np.clip(ct_slice, ct_range[0], ct_range[1])

    body_mask = body_mask.astype(float)
    body_mask[body_mask == 0] = np.nan

    fig, ax = plt.subplots()
    plt.axis('off')

    ax.imshow(
        ct_slice,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=ct_range[0], vmax=ct_range[1]),
        alpha=0.8)
    ax.imshow(
        body_mask,
        interpolation='none',
        cmap='plasma',
        norm=colors.Normalize(vmin=0, vmax=1),
        alpha=0.5
    )

    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def save_png_bcomp_overlay(ct_slice, ct_range, bcomp_mask, out_png):
    ct_slice = np.clip(ct_slice, ct_range[0], ct_range[1])
    bcomp_mask = bcomp_mask.astype(float)
    bcomp_mask[bcomp_mask <= 0] = np.nan

    cmap = colors.ListedColormap(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#ffff00'])
    boundaries = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    fig, ax = plt.subplots()
    plt.axis('off')

    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(
        ct_slice,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=ct_range[0], vmax=ct_range[1]),
        alpha=0.8)
    ax.imshow(
        bcomp_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )

    # print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


# def save_png_two_stage_combined_stack(raw_stack, process_stack, pconv_stack, bcomp_stack, out_png):
def save_png_two_stage_combined_stack(img_stack_dict, out_png):
    """
    :param img_stack_dict: 1) raw; 2) ct; 3) predict; 4) bcomp
    :param out_png:
    :return:
    """
    img_row_list = []
    for ds_name in ['raw', 'ct', 'predict', 'predict']:
        img_row_list.append(
            np.concatenate(
                [img_stack_dict[ds_name][idx_slice, :, :] for idx_slice in range(img_stack_dict[ds_name].shape[0])],
                axis=1
            )
        )
    img = np.concatenate(img_row_list, axis=0)

    null_mask = np.zeros(img_stack_dict['bcomp'].shape[1:])
    null_row_mask = np.concatenate([null_mask] * img_stack_dict['bcomp'].shape[0], axis=1)
    mask_row_list = [null_row_mask] * 3
    mask_row_list.append(
        np.concatenate(
            [img_stack_dict['bcomp'][idx_slice, :, :] for idx_slice in range(img_stack_dict['bcomp'].shape[0])],
            axis=1
        )
    )
    mask = np.concatenate(
        mask_row_list,
        axis=0
    ).astype(float)

    mask[mask <= 0] = np.nan
    cmap = colors.ListedColormap(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#ffff00'])
    boundaries = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    fig, ax = plt.subplots()
    plt.axis('off')
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(
        img,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=-1, vmax=1),
        alpha=0.8)
    ax.imshow(
        mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )

    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()



