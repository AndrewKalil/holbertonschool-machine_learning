#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow.keras as K


def lenet5(X):
    """ builds a modified version of the LeNet-5 architecture using
      tensorflow

    Args:
        X is a tf.placeholder of shape (m, 28, 28, 1) containing the input
          images for the network
            m is the number of images
    """
    # implement He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    my_layer = K.layers.Conv2D(filters=6,
                               kernel_size=(5, 5),
                               padding='same',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(X)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    my_layer = K.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2))(my_layer)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    my_layer = K.layers.Conv2D(filters=16,
                               kernel_size=(5, 5),
                               padding='valid',
                               activation='relu',
                               kernel_initializer=initializer,
                               )(my_layer)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    my_layer = K.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  )(my_layer)

    # Flattening between conv and dense layers
    my_layer = K.layers.Flatten()(my_layer)

    # Fully connected layer with 120 nodes
    my_layer = K.layers.Dense(units=120,
                              activation='relu',
                              kernel_initializer=initializer,
                              )(my_layer)

    # Fully connected layer with 84 nodes
    my_layer = K.layers.Dense(units=84,
                              activation='relu',
                              kernel_initializer=initializer,
                              )(my_layer)

    # Fully connected softmax output layer with 10 nodes
    my_layer = K.layers.Dense(units=10,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)

    network = K.Model(inputs=X, outputs=my_layer)

    network.compile(optimizer=K.optimizers.Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return network
