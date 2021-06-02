import tensorflow as tf
tf.keras.backend.set_floatx('float16')
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
import numpy as np
import os

import tensorflow.keras.layers as layers


class gated_conv(tf.keras.layers.Layer):
    def __init__(self, cnum, ksize, stride=1, rate=1, padding='SAME', activation=tf.nn.elu, training=True):
        super(gated_conv, self).__init__()
        self.gateconv1 = layers.Conv3D(cnum, ksize, stride, padding, dilation_rate=rate, activation=activation)
        self.gated_mask = layers.Conv3D(cnum, ksize, stride, padding, dilation_rate=rate, activation='sigmoid')
        self.instancen = instance_norm_layer()
        
    def call(self, x_in):
        x = self.gateconv1(x_in)
        gm = self.gated_mask(x_in)    
        result = x * gm
        result = self.instancen(result)
        return result
    
class gated_deconv(tf.keras.layers.Layer):
    def __init__(self, cnum, ksize=4, padding='SAME', training=True):
        super(gated_deconv, self).__init__()
        self.doublesize = layers.Conv3DTranspose(cnum, 4, 2, padding=padding)
        self.gatedcon_d = gated_conv(cnum, ksize, 1, padding=padding, training=training)
        self.instancen = instance_norm_layer()
        
    def call(self, x):
        x = self.doublesize(x) #double size
        x = self.gatedcon_d(x)
        x = self.instancen(x)
        return x


def instance_norm_layer():
    return InstanceNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform")


def generator(features = 64, im_shape=[128,128,128,1]):
    cropped_im = tf.keras.layers.Input(im_shape)
    
    encode = [
        gated_conv(features, 4, 1), #128, 128, 128, 64
        gated_conv(features, 4, 2), #64, 64, 64, 64 Down sample
        gated_conv(features, 4, 1), #64, 64, 64, 64
        gated_conv(features*2, 4, 2), #32, 32, 32, 128 Down sample
        gated_conv(features*2, 4, 1), #32, 32, 32, 128
        gated_conv(features*4, 4, 2), #16, 16, 16, 256 Down sample
        gated_conv(features*4, 4, 1), #16, 16, 16, 256
        gated_conv(features*4, 4, 2), #8, 8, 8, 256 Down sample
    ]
    
    dialate = [
        gated_conv(features*4, 4, 1, 2), #8, 8, 8, 256
        gated_conv(features*4, 4, 1, 4), #8, 8, 8, 256
        gated_conv(features*4, 4, 1, 8), #8, 8, 8, 256
        gated_conv(features*4, 4, 1, 16), #8, 8, 8, 256
    ]
    
    decode = [
        gated_deconv(features*4, 4), #16, 16, 16, 256 Up sample
        gated_conv(features*4, 4, 1), #16, 16, 16, 256
        gated_deconv(features*2, 4), #32, 32, 32, 128 Up sample
        gated_conv(features*2, 4, 1), #32, 32, 32, 128
        gated_deconv(features, 4), #64, 64, 64, 64 Up sample
        gated_conv(features, 4, 1), #64, 64, 64, 64
    ]
    
    last = layers.Conv3DTranspose(1, 4, 2, padding='same', activation=tf.nn.tanh)
    
    ##Build U-net##
    
    x = cropped_im

    # Downsampling through the model
    skips = []
    for down in encode:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    
    #Dialated Convolutions
    for dia in dialate:
        x = dia(x)
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(decode, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    x = last(x) #tanh is from -1,1 but our data is from -0.5, 0.5

    return tf.keras.Model(inputs=cropped_im, outputs=x)
  

class dlayer(tf.keras.layers.Layer):
    def __init__(self, features, ksize=4, stride=2, padding='same', name=None):
        super(dlayer, self).__init__(name=name)
        self.conv3d_spec = SpectralNormalization(layers.Conv3D(features, ksize, stride, padding=padding))
        self.lerelu = layers.LeakyReLU()
    def call(self, x):
        x=self.conv3d_spec(x)
        x=self.lerelu(x)
        return x


def discriminator_with_localmask(features = 64, im_shape=[128,128,128,1]):
    
    inputs = tf.keras.layers.Input(im_shape, name='disc_input')
    targets = tf.keras.layers.Input(im_shape, name='disc_target')
    mask = tf.keras.layers.Input(im_shape, name='disc_mask')
    ones_mask = tf.ones_like(mask, name='ones_mask')
    
    x = layers.Concatenate(name='global_concat')([inputs, targets])
    inputs_masked = inputs*mask
    targets_masked = targets*mask
    x_mask = layers.Concatenate(name='local_concat')([inputs_masked, targets_masked])
    
    
    encode_local = [
        dlayer(features, name='dlayer_local_0'), #64, 64, 64, 64
        dlayer(features*2, name='dlayer_local_1'), #32, 32, 32, 128
        dlayer(features*4, name='dlayer_local_2'), #16, 16, 16, 256
        dlayer(features*4, name='dlayer_local_3'), #8, 8, 8, 256
        dlayer(features*4, name='dlayer_local_4'), #4, 4, 4, 256
        dlayer(features*4, 4, 1, 'valid', name='dlayer_local_5'), #1, 1, 1, 256
        layers.Flatten(),
        SpectralNormalization(layers.Dense(256, use_bias=False)),
    ]
    
    encode_global = [
        dlayer(features, name='dlayer_global_0'), #64, 64, 64, 64
        dlayer(features*2, name='dlayer_global_1'), #32, 32, 32, 128
        dlayer(features*4, name='dlayer_global_2'), #16, 16, 16, 256
        dlayer(features*4, name='dlayer_global_3'), #8, 8, 8, 256
        dlayer(features*4, name='dlayer_global_4'), #4, 4, 4, 256
        dlayer(features*4, 4, 1, 'valid', name='dlayer_global_5'), #1, 1, 1, 256
        layers.Flatten(),
        SpectralNormalization(layers.Dense(256, use_bias=False)),
    ]
    
    for enc_l in encode_local:
        x_mask = enc_l(x_mask)
        
    for enc_g in encode_global:
        x = enc_g(x)
        
    x_out = layers.Concatenate()([x, x_mask])
    x_out = layers.LeakyReLU()(x_out)
    x_out = SpectralNormalization(layers.Dense(1, use_bias=False))(x_out)
    
    Disc_model = tf.keras.Model(inputs=[inputs,targets,mask], outputs=x_out)
    
    Disc_local_layer_names = ['dlayer_local_%d' % n for n in range(6)]
    Disc_local_layers = [Disc_model.get_layer(name).output for name in Disc_local_layer_names]
    

    return tf.keras.models.Model(inputs=[inputs,targets,mask], outputs=[x_out, Disc_local_layers])

def load_vgg():
    Vggpath = '../data/wbir_vggtrue3d.h5'
    Vgg_model = tf.keras.models.load_model(Vggpath)
    Vgg_model.trainable = False
    layer_outputs = [layer.output for layer in Vgg_model.layers[:8]] # Extracts the outputs of the top 7 layers
    layer_outputs = [layer_outputs[i] for i in [1,2,4,5,7]] #only get conv layers
    activation_model = tf.keras.models.Model(inputs=Vgg_model.input, outputs=layer_outputs)
    return activation_model

def get_intermediate_layers(model, which_layers):
    layer_outputs = [layer.output for layer in model.layers] # Extracts the outputs of the top 7 layers
    layer_outputs = [layer_outputs[i] for i in which_layers] #only get conv layers
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    return activation_model