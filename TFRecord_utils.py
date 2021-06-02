#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:04:15 2020

based on https://gist.github.com/CihanSoylu/02117f2d77136baf41ddb46789d8a331
@author: elizabeth
"""

import glob
import os
import sys
import numpy as np
import tensorflow as tf
 

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(volume, volume_shape):
    feature = {
        'uncropped_vol': _bytes_feature(volume),
        'height': _int64_feature(volume_shape[0]),
        'width': _int64_feature(volume_shape[1]),
        'depth': _int64_feature(volume_shape[2]),
        }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def read_tfrecord(serialized_example):
    feature_description = {
        'uncropped_vol': tf.io.FixedLenFeature((), tf.string),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
        }
    
    example = tf.io.parse_single_example(serialized_example, feature_description)
    uncropped_vol = tf.io.parse_tensor(example['uncropped_vol'], out_type = tf.float16)
    volume_shape = [example['height'], example['width'], example['depth']]
    uncropped_vol = tf.reshape(uncropped_vol, volume_shape)
    uncropped_vol = tf.expand_dims(uncropped_vol, -1)
    return uncropped_vol

def serialize_test_example(vol, body, volume_shape):
    #This is used for TFrecords of the PETCT dataset.  For when you actually are
    #trying to compensate for scan length
    feature = {
    'cropped_vol': _bytes_feature(vol),
    'cropped_bodymask': _bytes_feature(body),
    'height': _int64_feature(volume_shape[0]),
    'width': _int64_feature(volume_shape[1]),
    'depth': _int64_feature(volume_shape[2]),
    }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



def create_tfrdataset(PATH, tfrecord_file):
    '''
    PATH is path to folder containing .npz files of 3D image volumes.  One .npz file per volume
    tfrecord_file is the full filename of the desired tfrecord file you'd like to fill with your dataset
    
    This function takes in a your .npz image volumes and transforms them into a single Tensorflow record file.  
    This file will be used to load your database during training and testing stages.
    '''
    
    if not tfrecord_file.endswith('.tfrecord'):
        print('user provided tfrecord file name must end with ".tfrecord" ')
        sys.exit()
    
    train_files = glob.glob(os.path.join(PATH, '*.npz'))
    
    bigdata = np.zeros((len(train_files), 128, 128, 128))
    bigdata = bigdata.astype('float16')
    
    index = 0
    for file in train_files:
        decomp = np.load(file)['vol_data']
        decomp[decomp<0.02] = 0.0
        decomp = decomp.astype('float16')
        bigdata[index] = decomp
        print('loading file %d' % index)
        index += 1
    
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        volume_shape = bigdata[0].shape
        for entry in bigdata:
            example = serialize_example(tf.io.serialize_tensor(entry), volume_shape)
            writer.write(example)




