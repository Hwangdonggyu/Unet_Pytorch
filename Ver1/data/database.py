import numpy as np
import os
from .dataloader import BaseDataProvider
import nibabel as nib
import glob
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.measure import label 


class DataProvider(BaseDataProvider):
    def __init__(self, path, mode=None):
        super(DataProvider, self).__init__()
        self.path = path
        self.mode = mode
        self.data_idx = -1
        self.n_data = self._load_data()

    def _load_data(self):
        self.imageNum = []

        dataList = sorted(glob.glob(os.path.join(self.path, 'volume-*.nii')))
        maskList = sorted(glob.glob(os.path.join(self.path, 'segmentation-*.nii')))

        if len(dataList) != len(maskList):
            raise ValueError("The number of CTs and Labels is different.")
        
        paired_list = list(zip(dataList,maskList))

        if self.mode == 'train':
            paired_list = paired_list[:int(0.8 * len(paired_list))]
        else:
            paired_list = paired_list[:int(0.8 * len(paired_list)):]
        
        for vol_path, seg_path in paired_list:
            vol = nib.load(vol_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()
    
            if vol.shape != seg.shape:
                raise ValueError(f"Shape mismatch: {vol_path} vs {seg_path}")
            
            for i in range(vol.shape[2]):
                vol_slice = vol[:, :, i]
                seg_slice = seg[:, :, i]

                if np.any(seg_slice):
                    self.imageNum.append((vol_slice, seg_slice, i))
        
        if self.mode == 'train':
            np.random.shuffle(self.imageNum)
        
        return len(self.imageNum)

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.n_data:
            self.data_idx = 0
            if self.mode == 'train':
                np.random.shuffle(self.imageNum)
    
    def _next_data(self):
        self._shuffle_data_index()
        dataPath = self.imageNum[self.data_idx]

        data_vol = dataPath[0]
        data_seg = dataPath[1]
        data_index = dataPath[2]

        data_vol = np.clip(data_vol, 54, 66)

        # resize (필요시)
        data_vol = resize(data_vol,(256, 256), preserve_range=True)
        data_seg = resize(data_seg,(256, 256),  preserve_range=True)

        data_vol = (data_vol - 54) / (66 - 54)

        return data_vol, data_seg, data_index
    
    # unet 방식의 augment로 바꾸기
    def _augment_data(self, data, label):
        if self.mode == "train":
            # Flip horizon / vertical
            op = np.random.randint(0, 3)
            if op < 2:
                data, label = np.flip(data, op), np.flip(label, op)

        return data, label