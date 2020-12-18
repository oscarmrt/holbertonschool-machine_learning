#!/usr/bin/env python3
"""Program that creates and trains a gensim
word2vec model"""


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Function that creates and trains a gensim word2vec model"""
    
