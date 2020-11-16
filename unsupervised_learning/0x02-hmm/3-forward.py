#!/usr/bin/env python3
"""Program that performs the forward algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Function that performs the forward algorithm for
    a hidden markov model"""
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape != (Emission.shape[0], Emission.shape[0]):
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])
    for t in range(1, T):
        F[:, t] = np.multiply(Emission[:, Observation[t]],
                              np.dot(Transition.T, F[:, t - 1]))
    P = F[:, T - 1].sum()
    return P, F
