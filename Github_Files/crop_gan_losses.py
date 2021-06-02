#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:28:16 2020

https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py

@author: elizabeth
"""
import tensorflow as tf
import tensorflow.keras.backend as K

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class Attention_Mask(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention_Mask, self).__init__(dtype=tf.float32)
        self.avgpool = tf.keras.layers.AveragePooling3D(strides=2, dtype=tf.float32)
    
    def call(self, pred_img, tar_img):
        
        M_error = (pred_img - tar_img)**2
        M_error_min = tf.reduce_min(M_error)
        M_error_max = tf.reduce_max(M_error)
    
        M_guidance = (M_error - M_error_min) / (M_error_max - M_error_min + 1e-07) #float32
    
        M_guidance_l = []
        M_l = M_guidance
    
        M_guidance_l.append(M_l) #128
        M_guidance_l.append(M_l) #128
        M_l = self.avgpool(M_guidance_l[-1])
        M_guidance_l.append(M_l) #64
        M_guidance_l.append(M_l) #64
        M_l = self.avgpool(M_guidance_l[-1])
        M_guidance_l.append(M_l) #32
        
        return M_guidance_l

@tf.function
def Attention_Mask_function(pred_img, tar_img):
    M = Attention_Mask()(pred_img, tar_img)
    return M
    
def VGG_loss(act_pred, act_tar, pred_img, tar_img):
    """
    Based on VGG feature matching loss from:
    Hui Z, Li J, Wang X, Gao X. Image fine-grained inpainting. arXiv. 2020;(2):1-11.
    """

    M_guidance_l = Attention_Mask_function(pred_img, tar_img)
    w_l = []
    psi_diff = []
    N_psi = []
    
    
    for l in range(5):
        w_l.append((1e6) / (act_tar[l].shape[-1])**2) #changed from 1e3 to 1e6
        psi_diff.append(act_tar[l] - act_pred[l])
        N_psi.append(tf.size(act_tar[l], out_type=tf.dtypes.float32))
        
    
    #Self-guided loss
    L_selfguided_is = []
    for l in range(2):
        L_selfguided_i = (w_l[l] * tf.reduce_sum(tf.math.abs(M_guidance_l[l] * psi_diff[l]))) / N_psi[l]
        L_selfguided_is.append(L_selfguided_i)
    
    L_selfguided = tf.math.add_n(L_selfguided_is)
    
    #Feature matching loss
    L_fm_vggs = []
    for l in range(5):       
        L_fm_i = (w_l[l] * tf.reduce_sum(tf.math.abs(psi_diff[l]))) / N_psi[l]
        L_fm_vggs.append(L_fm_i)
        
    L_fm_vgg = tf.math.add_n(L_fm_vggs)
    
    return tf.cast(L_selfguided + L_fm_vgg, tf.float16)
    
    
def fm_dis_loss(target_layers, gen_layers):
    L_fm_dis_l = []
    for l in range(len(target_layers)):
        w_l = (1e3) / (target_layers[l].shape[-1])**2
        N_D = tf.size(target_layers[l], out_type=tf.dtypes.float32)
        D_diff = tf.reduce_sum(tf.math.abs(tf.cast(target_layers[l], tf.float32) - tf.cast(gen_layers[l], tf.float32)))
        L_fm_dis_l.append((w_l * D_diff) / N_D)
    
    L_fm_dis = tf.math.add_n(L_fm_dis_l)
    return tf.cast(L_fm_dis, tf.float16)


def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    
    #did the generator fool the discriminator?
    gan_loss = generator_wasshinge_loss(disc_generated_output) #try using wasserstein hinge loss
    
    #is generated image similar to target
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) 
    
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

def generator_loss_with_features(disc_generated_output, gen_output, target, tar_vgg, gen_vgg, tar_disc_act, gen_disc_act):
    
    gan_loss = generator_wasshinge_loss(disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    vgg_loss = VGG_loss(gen_vgg, tar_vgg, tf.cast(gen_output, tf.float32), tf.cast(target, tf.float32))
    fm_loss = fm_dis_loss(tar_disc_act, gen_disc_act)
    
    #These weights were empircally chosen through many trials
    gen_total_loss = (gan_loss) + (20 * l1_loss) + (10 * vgg_loss) + (5*fm_loss)
    
    return gen_total_loss, gan_loss, l1_loss, vgg_loss, fm_loss
    


def discriminator_loss(disc_real_output, disc_generated_output):
    
    #did the discriminator classify real input as real (ones)?
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    
    #did the discriminator classify generated input as fake (zeros)?
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

def discriminator_wasshinge_loss(disc_real_output, disc_generated_output, real_weights=1.0, generated_weights=1.0):
    hinged_real = tf.nn.relu(1.0 - disc_real_output)
    hinged_gen = tf.nn.relu(1.0 + disc_generated_output)

    loss_on_real = tf.math.reduce_mean(hinged_real)
    loss_on_generated = tf.math.reduce_mean(hinged_gen)

    discw_loss = loss_on_real + loss_on_generated
    
    return discw_loss, loss_on_generated, loss_on_real #added last two for debugging

def generator_wasshinge_loss(disc_generated_output, weights=1.0):
    genw_loss = - disc_generated_output
    genw_loss = tf.reduce_mean(genw_loss)
    
    return genw_loss

