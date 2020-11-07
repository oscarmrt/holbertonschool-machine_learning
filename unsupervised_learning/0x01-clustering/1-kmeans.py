#!/usr/bin/env python3
"""Program that performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Function that performs K-means on a dataset"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k < 1:
        return None
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    centroids = np.random.uniform(low=min, high=max, size=(k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=-1), axis=-1)
        newCentroid = np.copy(centroids)
        for c in range(k):
            if c not in clss:
                newCentroid[c] = np.random.uniform(min, max)
            else:
                newCentroid[c] = np.mean(X[clss == c], axis=0)
        if np.array_equal(newCentroid, centroids):
            return (centroids, clss)
        else:
            Centroids = newCentroid
    clss = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=-1), axis=-1)
    return (centroids, clss)
