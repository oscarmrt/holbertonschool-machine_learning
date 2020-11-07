#!/usr/bin/env python3
"""Program that calculates a GMM from a dataset"""
import sklearn.mixture


def gmm(X, k):
    """Function that calculates a GMM from a dataset"""
    g = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)
    return pi, m, S, clss, bic
