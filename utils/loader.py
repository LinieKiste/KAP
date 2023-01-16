from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F

import torch
import numpy as np
import pandas as pd
import pydicom
import os
import math

# PyTorch does not support uint16, so ToTensor() does not work :(
def default_transform_2d(input: np.ndarray):
    input = np.array(input)

    input = torch.tensor(input)
    input = F.convert_image_dtype(input, torch.float32)

    # add extra dimension
    input = torch.unsqueeze(input, 0)
    return input

def default_transform_3d(input: np.ndarray):
    input = np.array(input)
    input = torch.tensor(input)

    output = np.empty((len(input), len(input[0]), len(input[0][0])), dtype=np.float32)
    for i, img in enumerate(input):
        output[i] = (F.convert_image_dtype(img, torch.float32))
    output = np.array(output)
    output = torch.tensor(output)

    return output

class DicomDataset(Dataset):
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

        # sorting is not correct, but at least consistent
        slices = sorted(os.listdir(self.im_list[dicom_idx]))
        path = f"{self.im_list[dicom_idx]}/{slices[slice_idx]}"
        img = pydicom.dcmread(path).pixel_array

        slices = sorted(os.listdir(self.gt_list[dicom_idx]))
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

