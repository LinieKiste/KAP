from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F

import PIL
import torch
import numpy as np
import pandas as pd
import pydicom
import os
import math
import regex as re
import random

DATA_TYPE = torch.float32
AUGMENT_TIMES = 8

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

def default_transform_3d(input: np.ndarray, crop_at: tuple[int, int], hflip: bool, vflip: bool, rotate: bool):
    input = window_normalize(input)

    input = torch.tensor(input)
    output = torch.empty((len(input), len(input[0])//2, len(input[0][0])//2), dtype=DATA_TYPE)
    for i, img in enumerate(input):
        img = F.crop(img, *crop_at, len(img)//2, len(img[0])//2)
        if hflip:
            img = F.hflip(img)
        if vflip:
            img = F.vflip(img)
        if rotate:
            img = torch.unsqueeze(img, 0)
            img = F.rotate(img, random.choice([90, -90]))
            img = torch.squeeze(img, 0)
        output[i] = img.to(DATA_TYPE)

    output = torch.unsqueeze(output, 0)
    return output

def target_transform_3d(input: np.ndarray, crop_at: tuple[int, int], hflip: bool, vflip: bool, rotate: bool):
    input = torch.tensor(input)

    output = torch.empty((len(input), len(input[0])//2, len(input[0][0])//2), dtype=DATA_TYPE)
    for i, img in enumerate(input):
        img = F.crop(img, *crop_at, len(img)//2, len(img[0])//2)
        if hflip:
            img = F.hflip(img)
        if vflip:
            img = F.vflip(img)
        if rotate:
            img = torch.unsqueeze(img, 0)
            img = F.rotate(img, random.choice([90, -90]))
            img = torch.squeeze(img, 0)
        output[i] = img.to(DATA_TYPE)

    output = torch.unsqueeze(output, 0)
    return output

def crop_index(index) -> tuple[int, int]:
    RANGE = (64, 192)
    random.seed(index)
    return (random.randint(*RANGE), random.randint(*RANGE))

class DicomDataset3D(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths

        self.shortest = 2147483647
        self.get_shortest_dicom()

    def __getitem__(self, index):
        augmentation_no = index % len(self.im_list)
        crop_at = crop_index(augmentation_no)
        index = index // AUGMENT_TIMES
        hflip = True if augmentation_no > AUGMENT_TIMES // 2 else False
        vflip = True if augmentation_no > AUGMENT_TIMES // 4 and augmentation_no < (AUGMENT_TIMES // 4) * 3 else False
        rotate = True if augmentation_no % 2 == 0 else False
        
        # sample
        img = np.ndarray(shape=(self.shortest, 512, 512))
        slices = sorted(os.listdir(self.im_list[index]), key=lambda f: int(re.sub(r'\D', '', f)))
        for i, sl in zip(range(self.shortest), slices):
            path = f"{self.im_list[index]}/{sl}"

            tmp = pydicom.dcmread(path)
            tmp.decompress()
            tmp.convert_pixel_data('pillow')
            img[i] = tmp.pixel_array

        # ground truth
        target = np.ndarray(shape=(self.shortest, 512, 512))
        slices = sorted(os.listdir(self.gt_list[index]), key=lambda f: int(re.sub(r'\D', '', f)))
        for i, sl in zip(range(self.shortest), slices):
            path = f"{self.gt_list[index]}/{sl}"
            tmp = pydicom.dcmread(path)
            target[i] = tmp.pixel_array/255

        img = default_transform_3d(img, crop_at, hflip, vflip, rotate)
        target = target_transform_3d(target, crop_at, hflip, vflip, rotate)

        return img, target

    def __len__(self):
        return len(self.im_list) * AUGMENT_TIMES

    def get_shortest_dicom(self):
        for im in self.im_list:
            curr = len(os.listdir(im))
            self.shortest = curr if curr < self.shortest else self.shortest
        # ensure it is divisible by 16, so upsampling results in the same dimensions
        self.shortest //= 2**4
        self.shortest *= 2**4
