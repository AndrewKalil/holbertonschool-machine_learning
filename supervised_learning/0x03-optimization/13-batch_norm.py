#!/usr/bin/env python3
""" Description """
import numpy as np
import tensorflow as tf


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network using
       batch normalization

    Args:
        Z is a numpy.ndarray of shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in Z
        gamma is a numpy.ndarray of shape (1, n) containing the
         scales used for batch normalization
        beta is a numpy.ndarray of shape (1, n) containing the offsets
         used for batch normalization
        epsilon is a small number used to avoid division by zero
    """
    β = beta
    γ = gamma
    ε = epsilon

    μ = Z.mean(0)
    σ = Z.std(0)
    σ2 = Z.std(0) ** 2

    z_normalized = (Z - μ) / ((σ2 + ε) ** (0.5))
    Ẑ = γ * z_normalized + β

    return Ẑ
