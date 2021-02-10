#!/usr/bin/env python3
"""Program that has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """Function that has the trained agent play an episode"""
    state = env.reset()
    done = False
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()
        if done is True:
            if reward == 1:
                print(reward)
                break
        state = new_state
    env.close()
    return reward
