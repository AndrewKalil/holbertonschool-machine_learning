#!/usr/bin/env python3
""" Keras """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library

    Args:
        nx is the number of input features to the network
        layers is a list containing the number of nodes in each layer of the
          network
        activations is a list containing the activation functions used for
          each layer of the network
        lambtha is the L2 regularization parameter
        keep_prob is the probability that a node will be kept for dropout
    """
    # input placeholder
    inputs = K.Input(shape=(nx,))

    # regularization scheme
    reg = K.regularizers.L1L2(l2=lambtha)

    my_layer = K.layers.Dense(units=layers[0],
                              activation=activations[0],
                              kernel_regularizer=reg,
                              input_shape=(nx,))(inputs)

    for i in range(1, len(layers)):
        my_layer = K.layers.Dropout(1 - keep_prob)(my_layer)
        my_layer = K.layers.Dense(units=layers[i],
                                  activation=activations[i],
                                  kernel_regularizer=reg,
                                  )(my_layer)

    model = K.Model(inputs=inputs, outputs=my_layer)

    return model
