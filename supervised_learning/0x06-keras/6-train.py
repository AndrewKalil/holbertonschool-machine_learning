#!/usr/bin/env python3
""" Keras """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent

    Args:
        network is the model to train
        data is a numpy.ndarray of shape (m, nx) containing the input
          data
        labels is a one-hot numpy.ndarray of shape (m, classes) containing
          the labels of data
        batch_size is the size of the batch used for mini-batch gradient
          descent
        epochs is the number of passes through data for mini-batch gradient
          descent
        verbose is a boolean that determines if output should be printed
          during training
        shuffle is a boolean that determines whether to shuffle the batches
          every epoch. Normally, it is a good idea to shuffle, but for
          reproducibility, we have chosen to set the default to False.
    """
    callbacks = []
    if (validation_data):
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callbacks.append(early_stopping)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callbacks)

    return history
