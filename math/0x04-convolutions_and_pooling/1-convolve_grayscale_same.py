#!/usr/bin/env python3
""" Convolution and Pooling """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a same convolution on grayscale images

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
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    kw, kh = kernel.shape[1], kernel.shape[0]

    # calculates padding
    ph = max(int((kh - 1) / 2), int(kh / 2))
    pw = max(int((kw - 1) / 2), int(kw / 2))

    # image padding
    padded_image = np.pad(images,
                          pad_width=((0, 0), (ph, ph), (pw, pw)),
                          mode='constant', constant_values=0)

    # this is will form the shape of the output image
    output = np.zeros((m, h, w))

    # Loop over every pixel of the output
    for x in range(h):
        for y in range(w):
            # element-wise multiplication of the kernel and the image
            output[:, y, x] = \
                (kernel * padded_image[:,
                                       y: y + kh,
                                       x: x + kw]).sum(axis=(1, 2))

    return output
