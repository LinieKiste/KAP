#!/bin/env python3

import os
import zipfile
from tqdm import tqdm

def extract(path):
    if not os.path.exists(path):
        with zipfile.ZipFile(f'{path}.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{"/".join(path.split("/")[:-1])}')

if 'data' not in os.listdir():
    print("No data directory!")
    quit()

alldirs = 'data/3Dircadb1'

with zipfile.ZipFile(f'{alldirs}.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

for folder in tqdm(os.listdir(alldirs)):
    if '.sql' in folder: continue

    dicom = f'{alldirs}/{folder}/PATIENT_DICOM'
    masks = f'{alldirs}/{folder}/MASKS_DICOM'
    extract(dicom)
    extract(masks)

