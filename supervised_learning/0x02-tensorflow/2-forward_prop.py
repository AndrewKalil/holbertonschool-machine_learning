#!/usr/bin/env python3
"""Forward propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network

    Args:
        x:  placeholder for the input data
        layer_sizes: list containing the number of nodes in
            each layer of the network
        activations: list containing the activation functions
            for each layer of the network
    """
    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
