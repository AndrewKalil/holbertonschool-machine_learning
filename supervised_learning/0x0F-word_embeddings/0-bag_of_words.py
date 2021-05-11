#!/usr/bin/env python3
""" Bag of Words """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Creates a one hot encoding for sentences

    Args:
        sentences: list of sentences to analyze
        vocab ([type], optional): list of the vocabulary words to
          use for the analysis

    Returns:
        tupple: contains the word embedding and features
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names()

    return embedding, features
