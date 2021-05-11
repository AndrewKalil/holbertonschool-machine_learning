#!/usr/bin/env python3
""" Extract Word2Vec """
from tensorflow import keras


def gensim_to_keras(model):
    """converts a gensim word2vec model to a keras Embedding layer

    Args:
        model: a trained gensim word2vec models

    Returns:
        trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
