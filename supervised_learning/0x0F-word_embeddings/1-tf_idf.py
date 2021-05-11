#!/usr/bin/env python3
""" TF-IDF """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Creates a matrix containing weight of words

    Args:
        sentences: list of sentences to analyze
        vocab ([type], optional): list of the vocabulary words to
          use for the analysis

    Returns:
        tupple: contains the word embedding and features
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names()

    return embedding, features
