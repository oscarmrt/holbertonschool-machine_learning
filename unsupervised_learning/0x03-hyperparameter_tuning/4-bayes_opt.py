#!/usr/bin/env python3
"""Program that calculates the next
best sample location"""
import numpy as np
GP = __import__('2-gp').GaussianProcess
from scipy.stats import norm


class BayesianOptimization():
    """Class that calculates the next
    best sample location"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
    xsi=0.01, minimize=True):
        """Class constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Public instance method that calculates the
        next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        if not self.minimize:
            X_next = np.amax(self.gp.Y)
            imp = (mu - X_next - self.xsi)
        if self.minimize:
            X_next = np.amin(self.gp.Y)
            imp = (X_next - mu - self.xsi)
        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return self.X_s[np.argmax(EI)], EI
