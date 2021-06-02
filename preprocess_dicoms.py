#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:42:42 2020

@author: elizabeth
"""


import numpy as np
from scipy import ndimage as ndi
import scipy
import os, sys
import SimpleITK as sitk
import argparse
import ants
from skimage.transform import resize
import glob
import re




def Preprocess(args):
    
    SourceDir = args.SourceDir
    first = args.first
    last = args.last
    out_dir = args.out_dir
    template_dir = args.template_dir

    #load template head and neck image
    TemplateDicomImage = load_dicom_itk(template_dir)
    TemplateImage = sitk.GetArrayFromImage(TemplateDicomImage)

    #set up static portions of rigid registration to template image
    fixed_img = ants.from_numpy(TemplateImage.astype('float32'))
    fixed_img.set_origin((-450, -256, -256))


    #location of training image volumes. Each volume needs to be in its own folder
    folders = os.listdir(SourceDir)

    #maybe you don't want to go through everything
    iterater = len(os.listdir(out_dir)) #start at 0 or where you left off
    counter = first

    #if nothing was specified for 'last' assume you want every volume in SourceDir
    if not last:
        last = len(folders)

    subset = folders[first:last]

    #main part of processing
    for image_path in subset:
        print('working on volume: %s. Volume %d of %d to %d (%d Images Being Processed).\n' % (image_path, counter, first, last, len(subset)))
        counter = counter + 1

        #load image as a numpy array
        try:
            DicomImage = load_dicom_itk(os.path.join(SourceDir, image_path))
        except:
            print('Could not load Dicom information.  Skipping to next volume.\n')
            continue

        image = sitk.GetArrayFromImage(DicomImage)

        #rescale the image so that each voxel is 1x1x1mm^3
        try:
            rescale_factor = DicomImage.GetSpacing()
            image = ndi.zoom(image, rescale_factor[::-1])
            
            #I wanted to not have super long scans in this project
            if image.shape[0] > 600:
                image = image[(image.shape[0]-600):, :, :]
        except RuntimeError as err:
            print('Could not rescale image. Error message: ', err)
            continue


        #rigidly register image to template
        print('registering volume %s\n' % image_path)
        moving_img = ants.from_numpy(image.astype('float32'))
        moving_shape = image.shape
        moving_img.set_origin((-moving_shape[0], (-moving_shape[1])/2, (-moving_shape[2])/2))
        ants_reg = ants.registration(fixed = fixed_img, moving = moving_img, type_of_transform = 'QuickRigid')
        aligned_image = ants.apply_transforms(fixed = fixed_img, moving = moving_img, transformlist = ants_reg['fwdtransforms'], defaultvalue = -1024)
        aligned_image = aligned_image.numpy()


        print('resizing...\n')
        volume = resize(aligned_image, (256, 256, 256))

        minind = volume < -1024 #minimum HU is set to -1024
        maxind = volume > 3000 #maximum HU is set to 3000
        volume[minind] = -1024
        volume[maxind] = 3000

        print('normalizing...\n')
        volume += 1024
        volume *= (1/4024) #since it is shifted to make 0 the min, normalize by 1024+3000

        path_parts = re.split("/", image_path)
        savename = path_parts[-1]
        #save the images and segmentations to .npz files.  Has to be named 'vol_data'.  Assuming no more than 9999 volumes here
        np.savez(os.path.join(out_dir, "%04d.npz" % iterater), vol_data=vol_data)

        print('saved data: count = %d\n' % iterater)
        iterater = iterater+1

def load_dicom_itk(path):
    """
    read the dicom volume using SimpleITK
    """
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    img_dicomnames = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(img_dicomnames)
    imageitk = reader.Execute()
    if imageitk.GetSpacing()[2] > 5:
        new_spacing = list(imageitk.GetSpacing())
        try:
            new_spacing[2] = float(reader.GetMetaData(2, '0018|0050'))
        except:
            new_spacing[2] = 1.0
            print("using default slice thickness of 1mm")
        imageitk.SetSpacing(new_spacing)
        
    return imageitk

if __name__ == '__main__':
    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--SourceDir", help="Folder with subfolders for each Dicom volume you wish to preprocess.  Remove background (outside body contour) prior to starting this process")
    parser.add_argument("--first", help="Index in SourceDir to start procecessing. Default is 0", type=int, default=0)
    parser.add_argument("--last", help="Index in SourceDir to end processing. Default is the last index (all folders)", type=int, default=None)
    parser.add_argument("--out_dir", help="Folder to save your processed images")
    parser.add_argument("--template_dir", help="Folder containing Template Image dicom files")
    

    args = parser.parse_args()
    
    Preprocess(args)
