#!/bin/env python3

import os
import zipfile
import dicom2nifti

if 'data' not in os.listdir():
    print("No data directory!")
    quit()

alldirs = '../../data/3Dircadb1'
output_folder = 'data/nifti'

for folder in os.listdir(alldirs):
    if '.sql' in folder: continue

    dicom_zip = f'{alldirs}/{folder}/PATIENT_DICOM.zip'
    dicom_directory = f'{alldirs}/{folder}'

    if not os.path.exists(dicom_directory):
        with zipfile.ZipFile(dicom_zip, 'r') as zip_ref:
            zip_ref.extractall(dicom_directory)

    os.mkdir(f'{output_folder}/{folder}')
    dicom2nifti.convert_directory(dicom_directory, f'{output_folder}/{folder}', compression=True, reorient=True)

