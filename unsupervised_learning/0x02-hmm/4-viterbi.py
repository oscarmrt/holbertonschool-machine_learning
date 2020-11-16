#!/usr/bin/env python3
"""Program that calculates the most likely sequence of hidden
states for a hidden markov model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Function that calculates the most likely sequence of hidden
    states for a hidden markov model"""
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return (None, None)
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2 or \
       Transition.shape[0] != Transition.shape[1]:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0] != Transition.shape[0] !=\
       Initial.shape[0]:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    d = np.zeros([N, T])
    f = np.empty([N, T], dtype=int)
    d[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])
    for t in range(1, T):
        for i in range(N):
            d[i, t] = np.max(d[:, t - 1] * Transition[:, i]) *\
                      Emission[i, Observation[t]]
            f[i, t] = np.argmax(d[:, t - 1] * Transition[:, i])
    path = np.zeros(T)
    path[T - 1] = np.argmax(d[:, T - 1])
    for i in range(T - 2, -1, -1):
        path[i] = f[int(path[i + 1]), i + 1]
    P = np.max(d[:, T - 1:], axis=0)[0]
    path = [int(i) for i in path]
    return (path, P)
