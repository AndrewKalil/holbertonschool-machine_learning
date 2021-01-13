#!/usr/bin/env python3
""" Description """
import numpy as np
import tensorflow as tf


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a variable using the gradient descent with
    momentum optimization algorithm

    Args:
        alpha: the learning rate
        beta1: the momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: the previous first moment of var
    """
    # Exponentially Weighted Averages
    v = beta1 * v + (1 - beta1) * grad

    # variable update
    var = var - alpha * v
    return var, v
