from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F

import PIL
import torch
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import os
import math
import regex as re
import random

DATA_TYPE = torch.float32
USE_NIFTI = False

# PyTorch does not support uint16, so ToTensor() does not work :(
def default_transform_2d(input):
    input = np.array(input)

    input = torch.tensor(input)
    input = F.convert_image_dtype(input, torch.bfloat16)

    # add extra dimension
    input = torch.unsqueeze(input, 0)
    return input

class DicomDataset2D(Dataset):
    def __init__(self, csv_path, transform=default_transform_2d):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths

        self.transform = transform

        self.shortest = math.inf
        self.get_shortest_dicom()

    def __getitem__(self, index):
        dicom_idx = index//self.shortest
        slice_idx = index%self.shortest

        slices = sorted(os.listdir(self.im_list[dicom_idx]), key=lambda f: int(re.sub(r'\D', '', f)))
        path = f"{self.im_list[dicom_idx]}/{slices[slice_idx]}"
        img = pydicom.dcmread(path).pixel_array

        slices = sorted(os.listdir(self.gt_list[dicom_idx]), key=lambda f: int(re.sub(r'\D', '', f)))
        path = f"{self.gt_list[dicom_idx]}/{slices[slice_idx]}"
        target = pydicom.dcmread(path).pixel_array

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.im_list) * self.shortest

    def get_shortest_dicom(self):
        for im in self.im_list:
            curr = len(os.listdir(im))
            self.shortest = curr if curr < self.shortest else self.shortest

def window_normalize(input: np.ndarray, window_center=40, window_width=400):
    min_hu = window_center - (window_width / 2)
    max_hu = window_center + (window_width / 2)

    input = np.clip(input, min_hu, max_hu)
    input = (input - min_hu) / (max_hu - min_hu)

    return input

def default_transform_3d(input: np.ndarray, is_target: bool, crop_at: tuple[int, int], crop_factor: int, hflip: bool, vflip: bool, rotation_angle: float):
    if not is_target:
        input = window_normalize(input)

    input = torch.tensor(input).permute(2,0,1) if USE_NIFTI else torch.tensor(input)
    output = torch.empty((len(input), len(input[0])//crop_factor, len(input[0][0])//crop_factor), dtype=DATA_TYPE)
    for i, img in enumerate(input):
        img = F.crop(img, *crop_at, len(img)//crop_factor, len(img[0])//crop_factor)
        if hflip:
            img = F.hflip(img)
        if vflip:
            img = F.vflip(img)
        img = torch.unsqueeze(img, 0)
        img = F.rotate(img, rotation_angle)
        img = torch.squeeze(img, 0)
        output[i] = img.to(DATA_TYPE)

    output = torch.unsqueeze(output, 0)
    return output

class DicomDataset3D(Dataset):
    def __init__(self, csv_path):
        self.AUGMENT_TIMES = 16
        self.crop_factor = 2
        if csv_path == 'data/validation.csv':
            self.AUGMENT_TIMES = 4
            self.crop_factor = 1
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths

        self.shortest = [2147483647,2147483647,2147483647]
        self.get_shortest_nifti() if USE_NIFTI else self.get_shortest_dicom()

    def __getitem__(self, index):
        augmentation_no = index % len(self.im_list)
        index = index // self.AUGMENT_TIMES
        hflip = True if augmentation_no > self.AUGMENT_TIMES // 2 else False
        vflip = True if augmentation_no > self.AUGMENT_TIMES // 4 and augmentation_no < (self.AUGMENT_TIMES // 4) * 3 else False
        rotation_angle = 0 if augmentation_no % 2 == 0 else random.choice([90, -90])
        
        # sample
        if USE_NIFTI:
            img = nib.load(self.im_list[index]).get_fdata()   
            offsets = (
                random.randint(0, img.shape[2]-self.shortest[0]),
                random.randint(0, img.shape[0]-self.shortest[1]),
                random.randint(0, img.shape[1]-self.shortest[2])
                )
            img = img[
                offsets[1]:offsets[1]+self.shortest[1],
                offsets[2]:offsets[2]+self.shortest[2],
                offsets[0]:offsets[0]+self.shortest[0]
                ]
        # dicom
        else:
            img = np.ndarray(shape=(self.shortest[0], 512, 512))
            slices = sorted(os.listdir(self.im_list[index]), key=lambda f: int(re.sub(r'\D', '', f)))
            z_offset = random.randint(0, len(slices)-self.shortest[0])
            for i, sl in enumerate(slices[z_offset:z_offset+self.shortest[0]]):
                path = f"{self.im_list[index]}/{sl}"

                tmp = pydicom.dcmread(path)
                tmp.decompress()
                tmp.convert_pixel_data('pillow')
                img[i] = tmp.pixel_array

        # ground truth
        if USE_NIFTI:
            target = nib.load(self.gt_list[index]).get_fdata()   
            div = np.max(target)
            target = target[
                offsets[1]:offsets[1]+self.shortest[1],
                offsets[2]:offsets[2]+self.shortest[2],
                offsets[0]:offsets[0]+self.shortest[0]
                ] /div
            target = np.round(target)
        else:
            target = np.ndarray(shape=(self.shortest[0], 512, 512))
            slices = sorted(os.listdir(self.gt_list[index]), key=lambda f: int(re.sub(r'\D', '', f)))
            for i, sl in enumerate(slices[z_offset:z_offset+self.shortest[0]]):
                path = f"{self.gt_list[index]}/{sl}"
                tmp = pydicom.dcmread(path)
                target[i] = tmp.pixel_array/255

        crop_range = [
            (0, self.shortest[1]-(self.shortest[1]/self.crop_factor)),
            (0, self.shortest[2]-(self.shortest[2]/self.crop_factor))
        ]
        crop_at = (random.randint(*crop_range[0]), random.randint(*crop_range[1]))
        # vflip, hflip, rotation_angle = False, False, 0 # removes augmentations
        img = default_transform_3d(img, False, crop_at, self.crop_factor, hflip, vflip, rotation_angle)
        target = default_transform_3d(target, True, crop_at, self.crop_factor, hflip, vflip, rotation_angle)

        return img, target

    def __len__(self):
        return len(self.im_list) * self.AUGMENT_TIMES

    def get_shortest_dicom(self):
        self.shortest[1] = 512
        self.shortest[2] = 512
        for im in self.im_list:
            curr = len(os.listdir(im))
            self.shortest[0] = curr if curr < self.shortest[0] else self.shortest[0]
        # ensure it is divisible by 16, so upsampling results in the same dimensions
        self.shortest = [x // 2**5 for x in self.shortest]
        self.shortest = [x * 2**5 for x in self.shortest]

    def get_shortest_nifti(self):
        for im_path in self.im_list:
            img = nib.load(im_path).get_fdata()
            curr = img.shape
            self.shortest[0] = curr[2] if curr[2] < self.shortest[0] else self.shortest[0]
            self.shortest[1] = curr[0] if curr[0] < self.shortest[1] else self.shortest[1]
            self.shortest[2] = curr[1] if curr[1] < self.shortest[2] else self.shortest[2]

        self.shortest = [x // 2**5 for x in self.shortest]
        self.shortest = [x * 2**5 for x in self.shortest]

    def crop_index(self) -> tuple[int, int]:
        RANGE = (0, 192)
        return (random.randint(*RANGE), random.randint(*RANGE))
