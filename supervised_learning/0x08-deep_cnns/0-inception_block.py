#!/usr/bin/env python3
""" Deep Convolutional Neural Networks """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """builds an inception block as described in Going Deeper
      with Convolutions (2014)

    Args:
        A_prev is the output from the previous layer
        filters is a tuple or list containing F1, F3R, F3,F5R, F5,
            FPP, respectively:
            F1 is the number of filters in the 1x1 convolution
            F3R is the number of filters in the 1x1 convolution
                before the 3x3 convolution
            F3 is the number of filters in the 3x3 convolution
            F5R is the number of filters in the 1x1 convolution
                before the 5x5 convolution
            F5 is the number of filters in the 5x5 convolution
            FPP is the number of filters in the 1x1 convolution
                after the max pooling
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # implement He et, al initializer for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    # conv 1x1
    my_layer = K.layers.Conv2D(filters=F1,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(A_prev)

    # conv 1x1 before the 3x3 convolution
    my_layer1 = K.layers.Conv2D(filters=F3R,
                                kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(A_prev)

    my_layer1 = K.layers.Conv2D(filters=F3,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(my_layer1)

    # Conv 1x1 before the 5x5 convolution
    my_layer2 = K.layers.Conv2D(filters=F5R,
                                kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(A_prev)

    # Conv 5x5
    my_layer2 = K.layers.Conv2D(filters=F5,
                                kernel_size=(5, 5),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(my_layer2)

    # Max pooling layer with kernels of shape 3x3 with 1x1 strides
    my_layer3 = K.layers.MaxPool2D(pool_size=(3, 3),
                                   padding='same',
                                   strides=(1, 1))(A_prev)

    # Convolutional layer 1x1 convolution after the max pooling
    my_layer3 = K.layers.Conv2D(filters=FPP,
                                kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=initializer,
                                )(my_layer3)

    output = K.layers.concatenate([my_layer, my_layer1, my_layer2, my_layer3])

    return output
