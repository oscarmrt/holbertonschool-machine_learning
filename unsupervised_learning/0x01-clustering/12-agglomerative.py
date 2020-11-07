#!/usr/bin/env python3
"""Program that performs agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Function that performs agglomerative clustering on a dataset"""
    h = scipy.cluster.hierarchy
    Z = h.linkage(X, 'ward')
    f = h.fcluster(Z, dist, 'distance')
    h.dendrogram(Z)
    plt.show()
    return f
