#!/usr/bin/env python3
""" Transfer Learning """

from numpy.core.numeric import True_
import tensorflow.keras as K
import numpy as np
from tensorflow.keras import optimizers


def load_dataset():
    """ loads cifar 10 dataset and generates Xs Ys sets """
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # convert from integers to floats and normalizing to range to range 0-1
    X_train = X_train.astype('float16')
    X_test = X_test.astype('float16')
    X_train /= 255
    X_test /= 255

    # one hot encoding target values
    Y_train = K.utils.to_categorical(Y_train, 10)
    Y_test = K.utils.to_categorical(Y_test, 10)

    return X_train, Y_train, X_test, Y_test


def create_cnn(Y_train):
    """ creates cnn model for cifar 10 dataset """
    vgg = K.applications.vgg16.VGG16(include_top=False,
                                     weights='imagenet',
                                     input_shape=(32, 32, 3),
                                     classes=Y_train.shape[1])

    # Freezing most of the input layers
    for lyr in vgg.layers:
        if (lyr.name[0:5] != 'block'):
            lyr.trainable = False

    model = K.Sequential()
    model.add(vgg)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(256,
                             activation='relu',
                             kernel_initializer='he_uniform'))
    model.add(K.layers.Dense(10, activation='softmax'))
    model.summary()

    return model


def compile_cnn(cnn):
    """ compiles a neural network """
    opt = K.optimizers.SGD(lr=0.001, momentum=0.9)
    cnn.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return cnn


def train_cnn(
    cnn, X_train, Y_train, X_test, Y_test, batch_size, epochs):
    """ trains a neural network """
    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    return cnn.fit(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch = X_train.shape[0] // batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test)
    )


# script should not run when file is imported
if __name__ != '__main__':
    batch_size = 50
    epochs = 50

    # load dataset
    X_train, Y_train, X_test, Y_test = load_dataset()

    # define model
    cnn = create_cnn(Y_train)

    # compile model
    cnn = compile_cnn(cnn)

    # train model
    history = train_cnn(
        cnn, X_train, Y_train, X_test, Y_test, batch_size, epochs
    )

    cnn.save('cifar10.h5')


def preprocess_data(X, Y):
    """
    pre-processes the data for your model
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing
      the CIFAR 10 data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR
      10 labels for X
    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p
