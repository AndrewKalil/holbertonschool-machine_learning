#!/usr/bin/env python3
import tensorflow.keras as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is a tuple of integers containing the dimensions of the
      model input
    hidden_layers is a list containing the number of filters for each
      convolutional layer in the encoder, respectively
        - the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the
      latent space representation
    """
    input_img = K.layers.Input(shape=input_dims)

    enc_conv1 = Conv2D(12, (3, 3), activation='relu',
                       padding='same')(input_img)
    enc_pool1 = MaxPooling2D((2, 2), padding='same')(enc_conv1)
    enc_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_pool1)
    enc_ouput = MaxPooling2D((4, 4), padding='same')(enc_conv2)

    encoder = K.models.Model(input_img, enc_ouput)

    input_dec = K.layers.Input(shape=latent_dims)

    dec_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_ouput)
    dec_upsample2 = UpSampling2D((4, 4))(dec_conv2)
    dec_conv3 = Conv2D(12, (3, 3), activation='relu')(dec_upsample2)
    dec_upsample3 = UpSampling2D((2, 2))(dec_conv3)
    dec_output = Conv2D(1, (3, 3), activation='sigmoid',
                        padding='same')(dec_upsample3)

    decoder = K.models.Model(input_img, dec_output)

    autoencoder = Model(input_img, dec_output)
    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
