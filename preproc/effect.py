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
        return signal + noise.astype(np.int16)


class Echo(Effect):

    MIN_VOLUME = 0.1

    def __init__(self, sample_rate, delay, decay, repeats, prolong, effect=None):
        super(Echo, self).__init__(effect)
        self.sample_rate = sample_rate
        self.delay = delay
        self.decay = decay
        self.repeats = repeats
        self.prolong = prolong

    def apply(self, signal):
        if self.effect is not None:
            signal = self.effect.apply(signal)

        samples_to_delay = int(self.delay * self.sample_rate)
        new_signal = copy_buffer(signal, samples_to_delay * self.repeats, self.prolong)
        decays = np.linspace(self.decay, Echo.MIN_VOLUME, self.repeats)
        for i in range(signal.shape[0]):
            for j in range(self.repeats):
                if i + (samples_to_delay * j) >= new_signal.shape[0] and not self.prolong:
                    break
                new_signal[i + (samples_to_delay * j)] += int(signal[i] * decays[j])
        return new_signal


def copy_buffer(original, added_length, prolong):
    if prolong:
        length = list(original.shape)
        length[0] += added_length
        new_signal = np.zeros(length, dtype=np.int16)
    else:
        new_signal = np.zeros(original.shape, dtype=np.int16)
    new_signal[:original.shape[0]] += original
    return new_signal
