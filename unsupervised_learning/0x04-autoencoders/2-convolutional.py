#!/usr/bin/env python3
""" Autoencoder with convolutional layers """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims is a tuple of integers containing the dimensions of the
      model input
    filters  is a list containing the number of filters for each
      convolutional layer in the encoder, respectively
        - the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the
      latent space representation
    """
    inputs = keras.Input(shape=input_dims)

    # first convolutional layer
    conv_layer = keras.layers.Conv2D(filters=filters[0],
                                     kernel_size=(3, 3),
                                     padding="same",
                                     activation='relu',
                                     )(inputs)
    max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                            padding='same')(conv_layer)

    # subsequent convolutional layers:
    for i in range(1, len(filters)):
        conv_layer = keras.layers.Conv2D(filters=filters[i],
                                         kernel_size=(3, 3),
                                         padding="same",
                                         activation='relu',
                                         )(max_pool_2d)
        max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                padding='same')(conv_layer)

    encoder = keras.Model(inputs=inputs, outputs=max_pool_2d)
    encoder.summary()

    # ************************************************************
    # DECODER
    last_filter = input_dims[-1]

    # input placeholder
    inputs_dec = keras.Input(shape=latent_dims)

    # first conv layer
    my_conv_layer_dec = keras.layers.Conv2D(filters=filters[-1],
                                            kernel_size=(3, 3),
                                            padding="same",
                                            activation='relu',
                                            )(inputs_dec)

    upsampling_lay = keras.layers.UpSampling2D(
        size=(2, 2))(my_conv_layer_dec)

    # subsequent conv layers:
    for i in range(len(filters) - 2, -1, -1):
        my_conv_layer_dec = keras.layers.Conv2D(filters=filters[i],
                                                kernel_size=(3, 3),
                                                padding="same",
                                                activation='relu'
                                                )(upsampling_lay)

        upsampling_lay = keras.layers.UpSampling2D(
            size=(2, 2))(my_conv_layer_dec)

    # second to last convolution
    my_conv_layer_dec = keras.layers.Conv2D(filters=filters[0],
                                            kernel_size=(3, 3),
                                            padding="valid",
                                            activation='relu'
                                            )(upsampling_lay)

    """
    upsampling_lay = keras.layers.UpSampling2D(
        size=(2, 2))(my_conv_layer_dec)
    """

    #  last conv layer in the decoder
    my_conv_layer_dec = keras.layers.Conv2D(filters=last_filter,
                                            kernel_size=(3, 3),
                                            padding="valid",
                                            activation='sigmoid'
                                            )(my_conv_layer_dec)

    decoder = keras.Model(inputs=inputs_dec, outputs=my_conv_layer_dec)
    decoder.summary()

    # ************************************************************
    # AUTOENCODER
    auto_bottleneck = encoder.layers[-1].output
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs, outputs=auto_output)

    # compilation
    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy')

    return encoder, decoder, auto
