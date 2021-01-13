#!/usr/bin/env python3
""" Description """
import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """ calculates the normalization (standardization)
        constants of a matrix
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
