#!/usr/bin/env python3
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden layer
      in the encoder, respectively
        - the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
      representation
    """
    hidden_layers_length = len(hidden_layers)

    # encoder
    encoded = x = keras.Input(shape=(input_dims, ))
    for i in range(hidden_layers_length):
        x = keras.layers.Dense(hidden_layers[i], activation="relu")(x)
    h = keras.layers.Dense(latent_dims, activation="relu")(x)
    encoder = keras.models.Model(inputs=encoded, outputs=h)

    # decoder
    decoded = y = keras.Input(shape=(latent_dims, ))
    for j in range((hidden_layers_length - 1), -1, -1):
        y = keras.layers.Dense(hidden_layers[j], activation="relu")(y)
    r = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.models.Model(inputs=decoded, outputs=r)

    # autoencoder
    inputs = keras.Input(shape=(input_dims, ))
    outputs = decoder(encoder(inputs))

    autoencoder = keras.models.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
