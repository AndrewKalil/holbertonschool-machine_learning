#!/usr/bin/env python3
""" Convolution and Pooling """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ performs a convolution on grayscale images with custom padding

    Args:
        images is a numpy.ndarray with shape (m, h, w, c) containing multiple
         images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
          for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
            the image should be padded with 0’s
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
    """
    w, h, m = images.shape[2], images.shape[1], images.shape[0]
    c = images.shape[3]
    kw, kh, nc = kernels.shape[1], kernels.shape[0], kernels.shape[3]
    sw, sh = stride[1], stride[0]

    # calculates padding
    ph, pw = 0, 0

    if padding == "same":
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if isinstance(padding, tuple):
        # Extract required padding
        ph = padding[0]
        pw = padding[1]

    # image padding
    padded_image = np.pad(images,
                          pad_width=((0, 0),
                                     (ph, ph),
                                     (pw, pw),
                                     (0, 0)),
                          mode='constant', constant_values=0)

    new_h = int(((h + 2 * ph - kh) / sh) + 1)
    new_w = int(((w + 2 * pw - kw) / sw) + 1)

    # this is will form the shape of the output image
    output = np.zeros((m, new_h, new_w, nc))

    # Loop over every pixel of the output
    for y in range(new_h):
        for x in range(new_w):
            # over every kernel
            for v in range(nc):
                # element-wise multiplication of the kernel and the image
                output[:, y, x, v] = \
                    (kernels[:, :, :, v] *
                     images[:,
                            y * sh: y * sh + kh,
                            x * sw: x * sw + kw,
                            :]).sum(axis=(1, 2, 3))
    return output
