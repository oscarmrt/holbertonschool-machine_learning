#!/usr/bin/env python3
"""Program that initializes the Q-table"""
import gym
import numpy as np


def q_init(env):
    """Function that initializes the Q-table"""
    act_space = env.action_space.n
    obs_space = env.observation_space.n
    q_table = np.zeros((obs_space, act_space))
    return q_table
