import torch
import os
import glob
import nibabel as nib
import numpy as np
from skimage.transform import resize

class LiTSDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode=None):
        super().__init__()
        self.path = path
        self.mode = mode
        self.image_2d_path = []

        dataList = sorted(glob.glob(os.path.join(path, 'volume-*.nii')))
        maskList = sorted(glob.glob(os.path.join(path, 'segmentation-*.nii')))

        if len(dataList) != len(maskList):
            raise ValueError("Mismatch in number of volumes and masks")

        paired_list = list(zip(dataList, maskList))
        if self.mode == 'train':
            paired_list = paired_list[:int(0.8 * len(paired_list))]
        else:
            paired_list = paired_list[int(0.8 * len(paired_list)):]

        for vol_path, seg_path in paired_list:
            seg = nib.load(seg_path).get_fdata().astype(np.float32)

            for idx in range(seg.shape[2]):
                seg_slice = seg[:, :, idx]

                if np.any(seg_slice):
                    self.image_2d_path.append((vol_path, seg_path, idx))

    def __len__(self):
        return len(self.image_2d_path)
    
    def __getitem__(self, idx):
        vol_path, seg_path, slice_num = self.image_2d_path[idx]

        vol_slice = nib.load(vol_path).dataobj[:, :, slice_num].astype(np.float32)
        seg_slice = nib.load(seg_path).dataobj[:, :, slice_num].astype(np.float32)

        vol_slice = np.clip(vol_slice, -100, 400)
        vol_slice = resize(vol_slice, (256, 256), preserve_range=True)
        seg_slice = resize(seg_slice, (256, 256), preserve_range=True)
        vol_slice = (vol_slice + 100) / (400 + 100)

        vol_slice = torch.tensor(vol_slice, dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)
        seg_slice = torch.tensor(seg_slice, dtype=torch.float32).unsqueeze(0)

        return vol_slice, seg_slice