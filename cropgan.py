#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:51:09 2020

@author: elizabeth_mckenzie
"""
import argparse
import tensorflow as tf
import os
import sys
import time
import numpy as np
import io
import datetime
import pytz
timezone = pytz.timezone("America/Los_Angeles")

from matplotlib import pyplot as plt
from crop_GAN_networks import generator as generatorNet
from crop_GAN_networks import discriminator_with_localmask as discriminatorNet
from crop_GAN_networks import load_vgg as vggNet
from crop_gan_losses import generator_loss_with_features
from crop_gan_losses import discriminator_wasshinge_loss as discriminator_loss
from TFRecord_utils import read_tfrecord
from crop_data import crop_data
from scipy.ndimage import affine_transform as af
from save_testingimgs_as_nifti import main as niftisave


def train(args):
    
    PATH = args.PATH
    PATH_test = args.PATH_test
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    BUFFER_SIZE = args.BUFFER_SIZE
    BATCH_SIZE = args.BATCH_SIZE
    IMG_WIDTH = args.IMG_WIDTH
    IMG_HEIGHT = args.IMG_HEIGHT
    IMG_DEPTH = args.IMG_DEPTH
    OUTPUT_CHANNELS = args.OUTPUT_CHANNELS
    EPOCHS = args.EPOCHS
    STEPS = args.STEPS
    GPU = args.GPU
    continue_training = args.continue_training
    checkpoint_restore_dir = args.checkpoint_restore_dir
    
    
    ###load datasets###
    tfrecord_dataset = tf.data.TFRecordDataset(PATH)
    parsed_data = tfrecord_dataset.map(read_tfrecord)
    
    tfrecord_test_ds = tf.data.TFRecordDataset(PATH_test)
    parsed_test_data = tfrecord_test_ds.map(read_tfrecord)
    
    parsed_data = parsed_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    train_dataset = parsed_data.map(crop_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(lambda i, t, m: (2.0*(i-0.5), 2.0*(t-0.5), 1.0-m))
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat() #((1,128,128,128),(1,128,128,128),(1,128,128,128)) <-- (cropped, uncropped, mask)
    
    test_dataset = parsed_test_data.map(crop_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(lambda i, t, m: (2.0*(i-0.5), 2.0*(t-0.5), 1.0-m)) #(cropped, uncropped, mask)
    test_dataset = test_dataset.batch(1)

    
    ###define optimizers###
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    
    ###prepare gpu usage###
    gpu = '/device:GPU:%d' % GPU
    with tf.device(gpu):
        
        ###load networks###
        
        generator = generatorNet()
        discriminator = discriminatorNet()
        Vgg = vggNet()
        
           
        ###Setup Checkpoints##
        
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         discriminator_optimizer=discriminator_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
        
        if continue_training:
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_restore_dir))
        
        ###Training loop###
        
        current_time = datetime.datetime.now(timezone).strftime("%Y_%m_%d--%H_%M_%S")
        summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + current_time + '/training')
        summary_val_writer = tf.summary.create_file_writer(log_dir + "fit/" + current_time + '/validation')
        tf.keras.callbacks.TensorBoard(log_dir + "fit/" + current_time + '/Gen').set_model(generator)
    
        trainds_iter = iter(train_dataset)
        
        for epoch in range(EPOCHS):
            start = time.time()
            
            print("Epoch: ",  epoch)
            
            #for each epoch iterate over subset of dataset
            for n in range(STEPS):
                (input_image, target, mask) = next(trainds_iter)
                
                #training step
                loss, _, _, _ = train_step(input_image, target, mask, epoch, generator, discriminator, Vgg, generator_optimizer, discriminator_optimizer, n, training=True)
            print()
            
            #log to Tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', loss[0], step=epoch)
                tf.summary.scalar('gen_gan_loss', loss[1], step=epoch)
                tf.summary.scalar('gen_l1_loss', loss[2], step=epoch)
                tf.summary.scalar('vgg_loss', loss[3], step=epoch)
                tf.summary.scalar('fm_loss', loss[4], step=epoch)
                tf.summary.scalar('disc_loss', loss[5], step=epoch)
                
           #log val data to Tensorboard
            for (input_image_val, target_val, mask_val) in test_dataset.take(1):
                #create val loss info
                loss_val, pred_img, gen_vgg, tar_vgg = train_step(input_image_val, target_val, mask_val, epoch, generator, discriminator, Vgg, generator_optimizer, discriminator_optimizer, n, training=False)
                
                #create summary image of results
                summary_images, summary_images_act = visualize_training_output(input_image_val, target_val, pred_img, gen_vgg, tar_vgg)
                
            with summary_val_writer.as_default():
                tf.summary.image("Validation visualization", summary_images, step=epoch)
                tf.summary.image("Validation Activation visualization", summary_images_act, step=epoch)
                tf.summary.scalar('gen_total_loss', loss_val[0], step=epoch)
                tf.summary.scalar('gen_gan_loss', loss_val[1], step=epoch)
                tf.summary.scalar('gen_l1_loss', loss_val[2], step=epoch)
                tf.summary.scalar('vgg_loss', loss_val[3], step=epoch)
                tf.summary.scalar('fm_loss', loss_val[4], step=epoch)
                tf.summary.scalar('disc_loss', loss_val[5], step=epoch)
                
            #save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                
            print('Time taken for epoch {} is {:.2f} sec\n'.format(epoch+1, time.time()-start))
        checkpoint.save(file_prefix = checkpoint_prefix)

def train_step(input_image, target, mask, epoch, generator, discriminator, vgg_net, generator_optimizer,
               discriminator_optimizer, n, training=True):
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        #output from generator
        gen_output = generator(input_image, training=training)
        
        #replace known region with input
        gen_output = (gen_output * mask) + (input_image * (1-mask)) #mask is 1 in background, 0 in known region
        
        
        #outputs from VGG model
        gen_vgg = vgg_net(gen_output)
        tar_vgg = vgg_net(target)
        
        #discriminator on real target image
        disc_real_output, target_disc_layers = discriminator([input_image, target, mask], training=training)
        print('disc real output: ', tf.math.reduce_mean(disc_real_output).numpy())
        
        #discriminator on fake target image
        disc_generated_output, gen_disc_layers = discriminator([input_image, gen_output, mask], training=training)
        print('disc gen output: ', tf.math.reduce_mean(disc_generated_output).numpy())
            
        #the losses from the generator
        gen_total_loss, gen_gan_loss, gen_l1_loss, vgg_loss, fm_loss = generator_loss_with_features(disc_generated_output, gen_output, target, tar_vgg, gen_vgg, target_disc_layers, gen_disc_layers)
        #the losses from the discriminator
        disc_loss, loss_on_generated, loss_on_real = discriminator_loss(disc_real_output, disc_generated_output)
        
    
    if training:
        #calculate the gradients of the loss 
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        
        #calculate the gradients of the loss 
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        #apply these gradients to the optimizers
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        
        #apply these gradients to the optimizers
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        print('Epoch {} Step {} -- gen_total_loss: {:.3f}\tgen_gan_loss: {:.3f}\tgen_l1_loss: {:.3f}\tgen_vgg_loss: {:.3f}\tdisc_fm_loss: {:.3f}\tdisc_loss: {:.3f}\tdisc_loss_real: {:.3f}\tdisc_loss_fake: {:.3f}'.format(epoch, n, gen_total_loss, gen_gan_loss, gen_l1_loss, vgg_loss, fm_loss, disc_loss,loss_on_real, loss_on_generated))
    
    return (gen_total_loss, gen_gan_loss, gen_l1_loss, vgg_loss, fm_loss,  disc_loss), gen_output, gen_vgg, tar_vgg

            
def visualize_training_output(input_image, target, pred_img, gen_vgg, tar_vgg):
        #visualize results on testing dataset
        example_input = (input_image[0,:,:,64,0].numpy() + 1.0) / 2.0 #cropped
        example_target = (target[0,:,:,64,0].numpy() + 1.0) / 2.0 #uncropped
        pred_img = tf.clip_by_value(pred_img, -1.0, 1.0)
        example_pred = (pred_img[0,:,:,64,0].numpy() + 1.0) / 2.0 #predicted completed image

        
        figure = image_grid(example_input, example_target, example_pred)
        summary_image = plot_to_image(figure)
        figure_act = layers_image_grid(gen_vgg, tar_vgg)        
        summary_image_act = plot_to_image(figure_act)
        return summary_image, summary_image_act
        
def image_grid(test_input, tar, pred):
    figure = plt.figure(figsize=(15,15))
        
    display_list = [test_input, tar, pred]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    
    for i in range(3):
        plt.subplot(1,3, i+1)
        plt.title(title[i])
        #print(display_list[i])
        plt.imshow(np.flip((display_list[i] * 255).astype('uint8')), cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        
    return figure

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def layers_image_grid(gen_vgg, tar_vgg):
    
    figure = plt.figure(figsize=(15,15))
    cols = len(tar_vgg)
    row = 2
    g_title = ['Predicted: layer %d' % i for i in range(len(gen_vgg))]
    t_title = ['Target: layer %d' % i for i in range(len(tar_vgg))]
    
    for i, g in enumerate(gen_vgg):
        channel_image = process_activations(g)
        plt.subplot(2,cols, i+1)
        plt.title(g_title[i])
        plt.imshow(np.flip(channel_image), cmap='viridis')
        plt.axis('off')
        
    for j, t in enumerate(tar_vgg):
        channel_image = process_activations(t)
        plt.subplot(2,cols, i+j+2)
        plt.title(t_title[j])
        plt.imshow(np.flip(channel_image), cmap='viridis')
        plt.axis('off')
        
    return figure
        
        
def process_activations(layer_activation):
    avg_activation = np.nanmean(layer_activation, -1)
    size = layer_activation.shape[1]
    depth = layer_activation.shape[3]
    channel_image = avg_activation[0, :, :, depth//2]
    channel_image -= channel_image.mean()
    channel_image /= channel_image.std()
    channel_image *= 64
    channel_image += 128
    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    
    return channel_image


def test(args)          
     
    cropped_savepath = args.cropped_savepath
    prediction_savepath = args.prediction_savepath
    uncropped_savepath = args.uncropped_savepath
    mask_savepath = args.mask_savepath
    save_paths = [prediction_savepath, cropped_savepath, uncropped_savepath, mask_savepath]
    
    GPU = args.GPU
    PATH_test = args.PATH_test
    checkpoint_restore_dir = args.checkpoint_restore_dir
    
    
    #load testing dataset
    print('loading test dataset')
    tfrecord_test_ds = tf.data.TFRecordDataset(Path_test)
    parsed_test_data = tfrecord_test_ds.map(read_tfrecord) #uncropped image
    test_dataset = parsed_test_data.map(crop_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(lambda i, t, m: (2.0*(i-0.5), 2.0*(t-0.5), 1.0-m)) #(cropped, uncropped, mask)
    test_dataset = test_dataset.batch(1)
    
    gpu = '/gpu:%d' % GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    with tf.device(gpu):

        ###load trained networks###
        print('loading saved networks')
        generator = generatorNet()
        checkpoint = tf.train.Checkpoint(generator=generator)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_restore_dir))

        #generate images
        for n, (input_image, target, mask) in test_dataset.enumerate():

            start = time.time()
            prediction = generator(input_image, training=False)
            print('Time taken for prediction {} is {:.2f} sec\n'.format(n, time.time()-start))
            prediction = (prediction * mask) + (input_image * (1-mask))
            
            prediction = prediction.numpy()[0,...,0] #synthetically uncropped
            input_img = input_image.numpy()[0,...,0] #cropped
            target_img = target.numpy()[0,...,0] #uncropped
            mask_img = mask.numpy()[0,...,0] #mask

            input_img = return_to_HU(input_img)
            prediction = return_to_HU(prediction)
            target_img = return_to_HU(target_img)
            
            data_to_save = [prediction, input_img, target_img, mask_img]

            for path, data in zip(save_paths, data_to_save):
                np.savez(os.path.join(path, "%04d.npz" % n), vol_data=data)

            print('saved test image %d' % n)

        
        niftisave(*save_paths)


def return_to_HU(data):
    '''
    Take the output of the network (range from -1,1) and transform it back into HU values (-1024, 3000)
    Next reorient to HFS for standard dicom orientation (must be only 3D (no channel/batch dim))
    '''
    data = (data * 0.5) + 0.5 #go from -1,1 to 0,1
    data = (data * 4024) - 1024 #go from 0,1 to -1024,3000
    
    m = np.array(((0,0,-1,128), (0,-1,0,128),(1,0,0,0),(0,0,0,1)))
    data = af(data.astype(float), m, cval=-1024.0) #transform image to HFS
    return data.astype(np.float16)
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    #General Arguments / Arguments for Training
    parser.add_argument("--PATH", help="Path to uncropped training dataset (.tfrecord)")
    parser.add_argument("--PATH_test", help="Path to uncropped testing dataset (.tfrecord)")
    parser.add_argument("--checkpoint_dir", help="Directory to save checkpoints", default='../output/checkpoints/')
    parser.add_argument("--log_dir", help="Directory to save checkpoints", default='../output/logs/')
    parser.add_argument("--BUFFER_SIZE", help="Buffer for Shuffle", default=500)
    parser.add_argument("--BATCH_SIZE", help="Batch size", default=2)
    parser.add_argument("--IMG_WIDTH", help="Image Width", default=128)
    parser.add_argument("--IMG_HEIGHT", help="Image Height", default=128)
    parser.add_argument("--IMG_DEPTH", help="Image Depth", default=128)
    parser.add_argument("--OUTPUT_CHANNELS", help="Image Channels (1=Grayscale)", default=1)
    parser.add_argument("--EPOCHS", help="Number of Epochs to train", default=1000)
    parser.add_argument("--STEPS", help="Number of training steps per epoch", default=100)
    parser.add_argument("--GPU", help="GPU Number to use.  Currently only supports using single GPU", default=0)
    parser.add_argument("--Training", help="Are you training (True) or Predicting (False)", default=True, type=bool)
        
    #Arguments for Continuing Training from a Checkpoint
    parser.add_argument("--continue_training", help="Are you continuing training from a previous checkpoint?", default=False, type=bool)
    parser.add_argument("--checkpoint_restore_dir", help="Path to directory of checkpoint to restore", default=None)
    
    #Arguments for Inference
    parser.add_argument("--cropped_savepath", help="Path where you want to save cropped images", default=None)
    parser.add_argument("--prediction_savepath", help="Path where you want to save synthetically uncropped images", default=None)
    parser.add_argument("--uncropped_savepath", help="Path where you want to save  uncropped images", default=None)
    parser.add_argument("--mask_savepath", help="Path where you want to save masks used to generate cropped images", default=None)
    
    args = parser.parse_args()
    
    if args.continue_training and checkpoint_restore_dir is None:
        print('need to provide checkpoint restore directory (checkpoint_restore_dir) if continuing training')
        sys.exit()
        
    
    if args.Training:
        
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        if not os.path.isdir(args.log_dir):
            os.mkdir(args.log_dir)
            
        train(args)
    
    else:
        
        if not os.path.isdir(args.cropped_savepath):
            os.mkdir(args.cropped_savepath)
        if not os.path.isdir(args.prediction_savepath):
            os.mkdir(args.prediction_savepath)
        if not os.path.isdir(args.uncropped_savepath):
            os.mkdir(args.uncropped_savepath)
        if not os.path.isdir(args.mask_savepath):
            os.mkdir(args.mask_savepath)

        test(args)