#!/usr/bin/env python3
""" Convolutional Neural Network """
import tensorflow as tf


def lenet5(x, y):
    """ builds a modified version of the LeNet-5 architecture using
      tensorflow

    Args:
        x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
          images for the network
            m is the number of images
        y is a tf.placeholder of shape (m, 10) containing the one-hot labels
          for the network
    """

    # implement He et. al initialization for the layer weights
    initializer = \
        tf.contrib.layers.variance_scaling_initializer()

    # Conv layers
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    layer0 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    layer1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer0)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    layer2 = tf.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding='valid',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              name='layer')(layer1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    layer3 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer2)

    # Flattening between conv and dense layers
    layer3 = tf.layers.Flatten()(layer3)

    # Fully connected (Dense) layers
    # Fully connected layer with 120 nodes
    layer4 = tf.layers.Dense(units=120,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')(layer3)

    # Fully connected layer with 84 nodes
    layer5 = tf.layers.Dense(units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=initializer,
                             name='layer')(layer4)

    # Fully connected softmax output layer with 10 nodes
    layer6 = tf.layers.Dense(units=10,
                             kernel_initializer=initializer,
                             name='layer')(layer5)

    # loss
    loss = tf.losses.softmax_cross_entropy(y, layer6)

    # prediction
    y_pred = tf.nn.softmax(layer6)

    # train_op
    train_op = tf.train.AdamOptimizer(name='Adam').minimize(loss)

    # accuracy
    # from one y_pred one_hot to tag
    y_pred_t = tf.argmax(y_pred, 1)
    # from y one_hot to tag
    y_t = tf.argmax(y, 1)
    # comparison vector between tags (TRUE/FALSE)
    equal = tf.equal(y_pred_t, y_t)
    # average hits
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))

    return y_pred, train_op, loss, acc
