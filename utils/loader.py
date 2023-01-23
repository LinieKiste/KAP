from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import CenterCrop

import torch
import numpy as np
import pandas as pd
import pydicom
import os
import math
import regex as re

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

def default_transform_3d(input: np.ndarray):
    input = torch.tensor(input)
    output = torch.empty((len(input), len(input[0])//2, len(input[0][0])//2), dtype=torch.bfloat16)
    for i, img in enumerate(input):
        img = CenterCrop((len(img)//2, len(img[0])//2))(img)
        output[i] = F.convert_image_dtype(img, torch.bfloat16)

    output = torch.unsqueeze(output, 0)
    return output

class DicomDataset3D(Dataset):
    def __init__(self, csv_path, transform=default_transform_3d):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.gt_paths

        self.transform = transform

        self.shortest = 2147483647
        self.get_shortest_dicom()

    def __getitem__(self, index):
        
        img = np.ndarray(shape=(self.shortest, 512, 512))
        slices = sorted(os.listdir(self.im_list[index]), key=lambda f: int(re.sub(r'\D', '', f)))
        for i, sl in zip(range(self.shortest), slices):
            path = f"{self.im_list[index]}/{sl}"
            img[i] = pydicom.dcmread(path).pixel_array

        target = np.ndarray(shape=(self.shortest, 512, 512))
        slices = sorted(os.listdir(self.gt_list[index]), key=lambda f: int(re.sub(r'\D', '', f)))
        for i, sl in zip(range(self.shortest), slices):
            path = f"{self.gt_list[index]}/{sl}"
            target[i] = pydicom.dcmread(path).pixel_array

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.im_list)

    def get_shortest_dicom(self):
        for im in self.im_list:
            curr = len(os.listdir(im))
            self.shortest = curr if curr < self.shortest else self.shortest
