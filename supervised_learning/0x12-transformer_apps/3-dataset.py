#!/usr/bin/env python3
""" loads and preps a dataset for machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Loads and preps a dataset
    """

    def __init__(self):
        """
        Class constructor
        """
        ds, info = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            with_info=True,
            as_supervised=True)

        self.data_train, self.data_valid = ds
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)

        def filter_max_length(x, y, max_length=max_len):
            """ filter by new max length """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        self.data_train = self.data_train.filter(filter_max_length)

        self.data_train = self.data_train.cache()

        self.data_train = self.data_train.shuffle(
            info.splits["train"].num_examples
        ).padded_batch(
            batch_size, padded_shapes=([None], [None])
        )

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=([None],
                                                                      [None]))

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens
        """
        enc_pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        enc_en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return enc_pt, enc_en

    def tf_encode(self, pt, en):
        """
        tensorflow wrapper for the encode instance method
        """
        result_pt, result_en = tf.py_function(self.encode, [pt, en],
                                              [tf.int64, tf.int64])

        result_pt.set_shape([None])

        result_en.set_shape([None])

        return result_pt, result_en
