# CropGAN
synthetically extend medical imges to compensate for differing scan lengths

*Preprocessing*
To preprocess your own Dicom files, use the "preprocess_dicoms.py" file.  This requires the python ants package which can be installed from here: https://github.com/ANTsX/ANTsPy

*Training*
To train on your own preprocessed data, first build a tensorflow record dataset using the function "create_tfrdataset" found in the file "TFRecord_utils.py".  You'll need a training and testing dataset as the code is currently written.  The code could easily be modified to only use one dataset.

The training requires the trained VGG network created by Brian Avants.  Any publication using this training must also cite the paper: 
Avants B, Greenblatt E, Hesterman J, Tustison N. Deep Volumetric Feature Encoding for Biomedical Images. Vol 12120 LNCS. Springer International Publishing; 2020. doi:10.1007/978-3-030-50120-4_9
The trained VGG network used in this study can be found at https://figshare.com/articles/dataset/pretrained_networks_for_deep_learning_applications/7246985 . Select the file named "wbir_vggtrue3d.h5"

Run the file "cropgan.py" with the Training option set to True. Fill in other arguments as necessary.

*Prediction*
Similar to training, a preprocessed dataset must first be created.
Run the file "cropgan.py" with the Training option set to False. Fill in other arguments as necessary.

*Caveats*
-This project was completed using Tensorflow version 2.4.0 and Tensorflow Addons 0.11.0-dev
-This code assumes that you preprocess the images to be of size 128,128,128.  This has been hardcoded in some parts of the code.
-You should set all voxels outside the body contour to be -1024 HU in the dicoms prior to preprocessing with the preprocess_dicoms.py file

*Citations*
If you use any part of this work in your research, please cite: McKenzie, EM, Tong, N, Ruan, D, Cao, M, Chin, RK, Sheng, K. Using neural networks to extend cropped medical images for deformable registration among images with differing scan extents. Med. Phys. 2021; 00: 1– 13. https://doi.org/10.1002/mp.15039

