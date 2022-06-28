import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import tensorflow as tf
import tensorflow.keras as keras

from pathlib import Path
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import plot_model
from typing import Tuple



def dummy_fctn_vae():
    print('I am a dumnmy function withing the `vae_lib` module')

def img_encoder(enc_input_shape: Tuple[int, int, int], latent_space_dim: int) -> Model:
    """Build encoder converting image of specific shape into two vectors (means and log_var)
    
    enc_input_shape:    tupleshape of the image (height, width, channels)
    latent_space_dim:   int
    
    """    
    inputs = keras.Input(shape=enc_input_shape, name='img_input_layer')
    
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu', name='conv_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
        
    x = layers.Conv2D(64, 3, 2, padding='same', activation='relu', name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)

    x = layers.Conv2D(64, 3, 2, padding='same', activation='relu', name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)

    x = layers.Conv2D(64, 3, 2, padding='same', activation='relu', name='conv_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)

    flatten = layers.Flatten()(x)

    means = layers.Dense(latent_space_dim, name='mean')(flatten)
    log_vars = layers.Dense(latent_space_dim, name='log_var')(flatten)

    model = tf.keras.Model(inputs, (means, log_vars), name="Img-Encoder")
    
    return model

def sampling_model(distribution_params:Tuple):
    """Returns z stochastic vector based on distribution_params = (mean, log_var)"""
    means, log_vars = distribution_params
    epsilon = tf.random.normal(shape=tf.shape(means), mean=0., stddev=1.)
    return means + tf.math.exp(log_vars / 2) * epsilon

def sampling(input_1,input_2):
    """Building a lambda layer as the z sampling layer"""
    means = keras.Input(shape=input_1, name='input_means')
    log_vars = keras.Input(shape=input_2, name='input_log_vars')
    z = layers.Lambda(sampling_model, name='z-sampled-vector')([means, log_vars])
    
    sampler = tf.keras.Model([means,log_vars], z,  name="Img-Sampler")
    return sampler

def img_decoder(latent_space_dim: int = 200) -> Model:
    """"""
    z = keras.Input(shape=(latent_space_dim, ), name='dec_z_input_layer')

    # output for the Dense layer must be compatible with the first conv shape, 
    # i.e. the last encoder conv: (8, 8, 64). 8 * 8 * 64 = 4096
    x = layers.Dense(4096, name='dense_1')(z)
    x = layers.Reshape((8, 8, 64), name='Reshape_Layer')(x)
   
    # Block-1
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', activation='relu', name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
  
    # Block-2
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', activation='relu', name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    
    # Block-3
    x = layers.Conv2DTranspose(64, 3, 2, padding='same', activation='relu', name='conv_transpose_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)

    # Block-4
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', activation='relu', name='conv_transpose_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    
    # Block-5
    # Using sigmoid as activation to force pixel values to be between 0 and 1
    img_rec = layers.Conv2DTranspose(3, 3, 2, padding='same', activation='sigmoid', name='conv_transpose_5')(x)

    model = tf.keras.Model(z, img_rec, name="Img-Decoder")
    return model

def vae_model(enc_input_shape: Tuple[int, int, int], latent_space_dim: int = 200, p2weights: Path = None) -> Model:
    """ """
    img_inputs = keras.Input(shape=enc_input_shape, name='img_input_layer')
    img_enc = img_encoder(enc_input_shape)
    img_latent_space_sampler = sampling(latent_space_dim, latent_space_dim)
    img_dec = img_decoder(latent_space_dim)
    
    means, log_vars = img_enc(img_inputs)
    z = img_latent_space_sampler([means, log_vars])
    img_output = img_dec(z)
    m_and_lv = layers.Concatenate(axis=-1, name='means-and-log-vars')([means, log_vars])

    model = tf.keras.Model(inputs=[img_inputs], outputs=[img_output, m_and_lv], name="VAE")
    if p2weights is not None:
        model.load_weights(p2weights)
    
    return model
