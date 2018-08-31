import numpy as np


class Effect(object):

    def __init__(self, effect):
        self.effect = effect

    def apply(self, signal):
        return signal


class Noise(Effect):
    def __init__(self, mu, sigma, effect=None):
        super(Noise, self).__init__(effect)
        self.mu = mu
        if sigma < 0:
            raise ValueError("Sigma cannot be negative.", sigma)
        self.sigma = sigma

    def apply(self, signal):
        if self.effect is not None:
            signal = self.effect.apply(signal)

        noise = np.random.normal(self.mu, self.sigma, signal.shape)
        return signal + noise

