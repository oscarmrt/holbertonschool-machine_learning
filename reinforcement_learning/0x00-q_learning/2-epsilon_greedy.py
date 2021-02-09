#!/usr/bin/env python3
"""Program that uses epsilon-greedy to determine the next action"""
import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Function that uses epsilon-greedy to
determine the next action"""
    if np.random.uniform(0, 1) < epsilon:
        act = np.argmax(Q[state, :])
    else:
        act = np.random.randint(0, 3, None)
    return act
