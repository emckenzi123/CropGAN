#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:59:48 2020

@author: elizabeth_mckenzie
"""

import nibabel as nib
import numpy as np
import os
import glob

def main(predicted_imgs, cropped_imgs, uncropped_imgs, masks):
    #save .npz dataset images as .nii.gz for testing
    pred_files = np.sort(glob.glob(os.path.join(predicted_imgs, '*.npz')))
    cropped_files = np.sort(glob.glob(os.path.join(cropped_imgs, '*.npz')))
    uncropped_files = np.sort(glob.glob(os.path.join(uncropped_imgs, '*.npz')))
    mask_files = np.sort(glob.glob(os.path.join(masks, '*.npz')))
    
    for file, file_original, file_target, file_mask in zip(pred_files, cropped_files, uncropped_files, mask_files):
        
        load_and_transform(predicted_imgs, file)
        load_and_transform(cropped_imgs, file_original)
        load_and_transform(uncropped_imgs, file_target)
        load_and_transform(masks, file_mask)
        print('saved %s as nifti file' % file)
        print('saved %s as nifti file' % file_original)
        

def load_and_transform(path, file):
    
    this_file = os.path.join(path, file)
    npvol = np.load(this_file)['vol_data'].astype('float32')
    img_nii = nib.Nifti1Image(npvol, np.eye(4))
    nib.save(img_nii, os.path.join(path, file.replace('.npz', '.nii.gz')))
    
if __name__ == '__main__':
    main()