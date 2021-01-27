#!/usr/bin/env python3
""" Convolution and Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images

    Args:
        images is a numpy.ndarray with shape (m, h, w, c) containing
          multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel_shape is a tuple of (kh, kw) containing the kernel shape
          for the pooling
            kh is the height of the kernel
            kw is the width of the kernel
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling
    """
    c, w, = images.shape[3], images.shape[2]
    h, m = images.shape[1], images.shape[0]
    kw, kh = kernel_shape[1], kernel_shape[0]
    sw, sh = stride[1], stride[0]

    new_h = int(((h - kh) / sh) + 1)
    new_w = int(((w - kw) / sw) + 1)

    # initialize convolution output tensor
    output = np.zeros((m, new_h, new_w, c))

    # Loop over every pixel of the output
    for x in range(new_w):
        for y in range(new_h):
            # element-wise multiplication of the kernel and the image
            if mode == 'max':
                output[:, y, x, :] = \
                    np.max(images[:,
                                  y * sh: y * sh + kh,
                                  x * sw: x * sw + kw], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = \
                    np.mean(images[:,
                                   y * sh: y * sh + kh,
                                   x * sw: x * sw + kw], axis=(1, 2))

    return output
