#!/usr/bin/env python3
""" contains the pca function"""
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset
    :param X: numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    :param var: fraction of the variance that the PCA
        transformation should maintain
    :return: weights matrix, W,
        that maintains var fraction of X‘s original variance
        W is a numpy.ndarray of shape (d, nd)
            where nd is the new dimensionality of the transformed X
    """
    u, s, vh = np.linalg.svd(X)

    cumsum = np.cumsum(s)

    dim = [i for i in range(len(s)) if cumsum[i] / cumsum[-1] >= var]
    ndim = dim[0] + 1

    return vh.T[:, :ndim]
