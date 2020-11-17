#!/usr/bin/env python3
"""Program that performs the backward algorithm
for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Function that performs the backward algorithm
    for a hidden markov model"""
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return (None, None)
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
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
    B = np.empty([N, T], dtype='float')
    B[:, T - 1] = 1
    for t in reversed(range(T - 1)):
        B[:, t] = np.dot(Transition,
                         np.multiply(Emission[:,
                                     Observation[t + 1]], B[:, t + 1]))
    P = np.dot(Initial.T, np.multiply(Emission[:, Observation[0]], B[:, 0]))
    return (P, B)
