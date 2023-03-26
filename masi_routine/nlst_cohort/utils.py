import nibabel as nib
import os


class ScanWrapper:
    def __init__(self, img_path):
        self._img = nib.load(img_path)
        self._path = img_path

    def get_path(self):
        return self._path

    def get_file_name(self):
        return os.path.basename(self._path)

    def get_header(self):
        return self._img.header

    def get_affine(self):
        return self._img.affine

    def get_shape(self):
        return self.get_header().get_data_shape()

    def get_number_voxel(self):
        return np.prod(self.get_shape())

    def get_data(self):
        return self._img.get_fdata()

    def get_voxel_size(self):
        return self._img.header.get_zooms()

    def get_center_slices(self):
        im_data = self.get_data()
        im_shape = im_data.shape
        slice_x = im_data[int(im_shape[0] / 2) - 1, :, :]
        slice_x = np.flip(slice_x, 0)
        slice_x = np.rot90(slice_x)
        slice_y = im_data[:, int(im_shape[0] / 2) - 1, :]
        slice_y = np.flip(slice_y, 0)
        slice_y = np.rot90(slice_y)
        slice_z = im_data[:, :, int(im_shape[2] / 2) - 1]
        slice_z = np.rot90(slice_z)

        return slice_x, slice_y, slice_z

    def save_scan_same_space(self, file_path, img_data):
        # logger.info(f'Saving image to {file_path}')
        img_obj = nib.Nifti1Image(img_data,
                                  affine=self.get_affine(),
                                  header=self.get_header())
        nib.save(img_obj, file_path)

    def save_scan_flat_img(self, data_flat, out_path):
        img_shape = self.get_shape()
        data_3d = convert_flat_2_3d(data_flat, img_shape)
        self.save_scan_same_space(out_path, data_3d)