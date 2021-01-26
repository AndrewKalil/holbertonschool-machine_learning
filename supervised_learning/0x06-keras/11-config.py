#!/usr/bin/env python3
""" Keras """

import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format
    :param network: model whose weights should be saved
    :param filename: path of the file that the model should be saved to
    :return: None
    """
    json_string = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_string)
    return None


def load_config(filename):
    """
     loads a model’s weights
    :param filename:  p path of the file containing the model’s
        configuration in JSON forma
    :return: loaded model
    """
    with open(filename, "r") as f:
        network_string = f.read()
    network = K.models.model_from_json(network_string)
    return network
