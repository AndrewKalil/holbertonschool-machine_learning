#!/usr/bin/env python3
""" This module has the create_masks(inputs, target) method """

import tensorflow.compat.v2 as tf


def create_padding_mask(seq):
    """ This method Mask all the pad tokens in the batch of sequence """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """ This method is used to mask the future tokens in a sequence"""
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_masks(inputs, target):
    """ This method creates all masks for training/validation """
    enc_padding_mask = create_padding_mask(inputs)
    dec_padding_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
