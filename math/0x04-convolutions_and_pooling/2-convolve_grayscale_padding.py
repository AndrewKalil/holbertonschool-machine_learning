#!/usr/bin/env python3
""" Convolution and Pooling """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a convolution on grayscale images with custom padding

    Args:
        images is a numpy.ndarray with shape (m, h, w) containing multiple
          grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
          for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
            the image should be padded with 0’s
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]

    # calculates padding
    ph = padding[0]
    pw = padding[1]

    # image padding
    padded_image = np.pad(images,
                          pad_width=((0, 0), (ph, ph), (pw, pw)),
                          mode='constant', constant_values=0)
    new_h = int(padded_image.shape[1] - kh + 1)
    new_w = int(padded_image.shape[2] - kw + 1)

    # this is will form the shape of the output image
    output = np.zeros((m, new_h, new_w))

    # Loop over every pixel of the output
    for x in range(new_w):
        for y in range(new_h):
            # element-wise multiplication of the kernel and the image
            output[:, y, x] = \
                (kernel * padded_image[:,
                                       y: y + kh,
                                       x: x + kw]).sum(axis=(1, 2))

    return output
