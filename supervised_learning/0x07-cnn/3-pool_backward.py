#!/usr/bin/env python3
""" Convolutional Neural Network """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network

    Args:
        dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
          the partial derivatives with respect to the output of the pooling
          layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
          the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
        kernel_shape is a tuple of (kh, kw) containing the size of the kernel
          for the pooling
            kh is the kernel height
            kw is the kernel width
        stride is a tuple of (sh, sw) containing the strides for the
          convolution
            sh is the stride for the height
            sw is the stride for the width
        mode is a string containing either max or avg, indicating whether
          to perform maximum or average pooling, respectively
    """

    # Retrieving dimensions from dA
    m, h_new, w_new, c_new = dA.shape

    # Retrieving dimensions from A_prev shape
    m, h_prev, w_prev, c_prev = A_prev.shape

    # Retrieving dimensions from kernel_shape
    kh, kw = kernel_shape

    # Retrieving stride
    sh, sw = stride

    # Initialize the output with zeros
    dA_prev = np.zeros_like(A_prev, dtype=dA.dtype)

    # Looping over vertical(h) and horizontal(w) axis of output volume
    for z in range(m):
        for y in range(h_new):
            for x in range(w_new):
                for v in range(c_new):
                    pool = A_prev[z, y * sh:(kh+y*sh), x * sw:(kw+x*sw), v]
                    dA_aux = dA[z, y, x, v]
                    if mode == 'max':
                        z_mask = np.zeros(kernel_shape)
                        _max = np.amax(pool)
                        np.place(z_mask, pool == _max, 1)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), v] += z_mask * dA_aux
                    if mode == 'avg':
                        avg = dA_aux / kh / kw
                        o_mask = np.ones(kernel_shape)
                        dA_prev[z, y * sh:(kh + y * sh),
                                x * sw:(kw+x*sw), v] += o_mask * avg
    return dA_prev
