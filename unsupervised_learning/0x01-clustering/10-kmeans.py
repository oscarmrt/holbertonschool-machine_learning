#!/usr/bin/env python3
"""Program that performs K-means on a dataset"""
import sklearn.cluster


def kmeans(X, k):
    """Function that performs K-means on a dataset"""
    C, clss, _ = sklearn.cluster.k_means(X, k)
    return C, clss
