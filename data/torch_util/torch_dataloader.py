import torch
import os
import glob
import nibabel as nib
import numpy as np
from skimage.transform import resize

class LiTSDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()
        self.path = path
        self.mode = mode
        self.image_2d = []

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
            vol = nib.load(vol_path).get_fdata().astype(np.float32)
            seg = nib.load(seg_path).get_fdata().astype(np.float32)

            for i in range(vol.shape[2]):
                vol_slice = vol[:, :, i]
                seg_slice = seg[:, :, i]

                if np.any(seg_slice):
                    vol_slice = np.clip(vol_slice, 54, 66)
                    vol_slice = resize(vol_slice, (256, 256), preserve_range=True)
                    seg_slice = resize(seg_slice, (256, 256), preserve_range=True)
                    vol_slice = (vol_slice - 54) / (66 - 54)

                    self.image_2d.append((vol_slice, seg_slice))

    def __len__(self):
        return len(self.image_2d)

    def __getitem__(self, idx):
        vol, seg = self.image_2d[idx]

        vol = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)  # (1, 256, 256)
        seg = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)

        return vol, seg
