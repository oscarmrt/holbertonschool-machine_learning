#!/usr/bin/env python3
"""Program that performs the Baum-Welch
algorithm for a hidden markov model"""
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Function that performs the Baum-Welch algorithm
    for a hidden markov model"""
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
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
    T = Observations.shape[0]
    M, N = Emission.shape
    for n in range(1, iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((M, M, T - 1))
        for i in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, i].T, Transition) *
                                 Emission[:, Observations[i + 1]].T,
                                 beta[:, i + 1])
            for j in range(M):
                numerator = alpha[j, i] * Transition[j] *\
                    Emission[:, Observations[i + 1]].T * beta[:, i + 1].T
                xi[j, :, i] = numerator / denominator
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        denominator = np.sum(gamma, axis=1)
        for i in range(N):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)
        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return (Transition, Emission)
