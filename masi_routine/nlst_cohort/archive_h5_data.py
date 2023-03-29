import glob
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import ScanWrapper
from joblib import Parallel, delayed


def filter_scan_w_condition(in_record_df):
    in_record_df = in_record_df.loc[
        (in_record_df['nii_file'] == 1) &
        (in_record_df['png_file'] == 1) &
        (in_record_df['num_img_dimension'] == 3) &
        (in_record_df['orient'] == 'LAS')]

    z_space_threshold = 2.5
    in_record_df = in_record_df.loc[in_record_df['voxel_size_z'] < z_space_threshold + 0.001]

    image_height_threshold = 200
    in_record_df = in_record_df.dropna(subset=['voxel_size_z', 'num_dcm'])
    in_record_df['z_height'] = in_record_df['voxel_size_z'] * in_record_df['num_dcm']
    in_record_df = in_record_df.loc[in_record_df['z_height'] >= image_height_threshold]

    return in_record_df


def get_num_candidate_cases():
    """
    We only work on the T0 data
    """
    print(f'Load {in_record_csv}')
    in_record_df = pd.read_csv(in_record_csv)

    n_subject = len(list(set(in_record_df['pid'].to_list())))
    n_series = len(in_record_df.index)
    print(f'Before filtering: subject - {n_subject}; series - {n_series}')

    in_record_df = filter_scan_w_condition(in_record_df)
    n_subject = len(list(set(in_record_df['pid'].to_list())))
    n_series = len(in_record_df.index)
    print(f'After filtering: subject - {n_subject}; series - {n_series}')


def generate_all_t0():
    output_dir = os.path.join(archive_root, 'all')
    os.makedirs(output_dir, exist_ok=True)

    in_record_df = filter_scan_w_condition(pd.read_csv(in_record_csv))
    # in_record_df = in_record_df.tail(2000)
    # use_record_df = in_record_df.head(10)
    uid_list = in_record_df['series_uid'].to_list()

    # print(uid_list)

    def _process_single_case(uid):
        in_nii = os.path.join(in_data_dir, f'{uid}.nii.gz')
        out_h5 = os.path.join(output_dir, f'{uid}.hdf5')
        if os.path.exists(out_h5):
            rm_cmd = f'rm -f {out_h5}'
            os.system(rm_cmd)

        img_obj = ScanWrapper(in_nii)
        img = img_obj.get_data()
        img_shape = img.shape
        voxel_size = img_obj.get_voxel_size()

        assert img_shape[0] == img_shape[1]

        db = h5py.File(out_h5, 'a')
        db.create_dataset(
            'ct',
            data=img,
            chunks=(img_shape[0], img_shape[1], 1),
            compression='gzip'
        )
        db.attrs['voxel_size'] = voxel_size
        db.close()

    Parallel(
        n_jobs=80,
        prefer='processes'
    )(delayed(_process_single_case)(uid)
      for uid in tqdm(uid_list, total=len(uid_list)))


def check_missing():
    output_dir = os.path.join(archive_root, 'all')
    in_record_df = filter_scan_w_condition(pd.read_csv(in_record_csv))
    in_record_df = in_record_df.tail(200)

    uid_list = in_record_df['series_uid'].to_list()

    n_missing = 0
    for uid in uid_list:
        hdf5_path = os.path.join(output_dir, f'{uid}.hdf5')
        if not os.path.exists(hdf5_path):
            print(f'Missing {uid}')
            n_missing += 1

    print(f'Total missing: {n_missing}')


if __name__ == '__main__':
    archive_root = '/local_storage2/Data/NLST/HDF5/T0'
    os.makedirs(archive_root, exist_ok=True)
    in_record_csv = os.path.join('/local_storage/xuk9/Data/NLST/NIfTI',
                                 f'T0_all.csv')
    in_data_dir = '/local_storage/xuk9/Data/NLST/NIfTI/T0_all'

    # get_num_candidate_cases()
    generate_all_t0()
    # check_missing()
