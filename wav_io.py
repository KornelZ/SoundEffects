import scipy.io.wavfile
import numpy as np


def read_wav(path):
    return scipy.io.wavfile.read(path)


def write_wav(path, data, sample_rate):
    data = data.astype(np.int16) #type int16 signed wav file
    scipy.io.wavfile.write(path, sample_rate, data)

