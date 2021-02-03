#!/usr/bin/env python3
""" Convolutional Neural Network """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional layer of a neural
      network

    Args:
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
          containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
          the kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the
          biases applied to the convolution
        activation is an activation function applied to the convolution
        padding is a string that is either same or valid, indicating the
          type of padding used
        stride is a tuple of (sh, sw) containing the strides for the
          convolution
            sh is the stride for the height
            sw is the stride for the width
    """

    # Retrieving dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieving dimensions from W's shape
    (kh, kw, c_prev, c_new) = W.shape

    # Retrieving stride
    (sh, sw) = stride

    # Setting padding for valid
    pw, ph = 0, 0

    # Compute the output dimensions
    n_h = int(((h_prev + 2 * ph - kh) / sh) + 1)
    n_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    # Setting padding for same
    if padding == 'same':
        if kh % 2 == 0:
            ph = int((h_prev * sh + kh - h_prev) / 2)
            n_h = int(((h_prev + 2 * ph - kh) / sh))
        else:
            ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
            n_h = int(((h_prev + 2 * ph - kh) / sh) + 1)

        if kw % 2 == 0:
            pw = int((w_prev * sw + kw - w_prev) / 2)
            n_w = int(((w_prev + 2 * pw - kw) / sw))
        else:
            pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
            n_w = int(((w_prev + 2 * pw - kw) / sw) + 1)

    # pad images
    images = np.pad(A_prev,
                    pad_width=((0, 0),
                               (ph, ph),
                               (pw, pw),
                               (0, 0)),
                    mode='constant', constant_values=0)

    # Initialize the output with zeros
    output = np.zeros((m, n_h, n_w, c_new))

    # Looping over vertical(h) and horizontal(w) axis of output volume
    for y in range(n_h):
        for x in range(n_w):
            # over every channel
            for v in range(c_new):
                # element-wise multiplication of the kernel and the image
                output[:, y, x, v] = \
                    (W[:, :, :, v] *
                     images[:,
                            y * sh: y * sh + kh,
                            x * sw: x * sw + kw,
                            :]).sum(axis=(1, 2, 3))

                output[:, y, x, v] = \
                    (activation(output[:, y, x, v] +
                                b[0, 0, 0, v]))

    return output
