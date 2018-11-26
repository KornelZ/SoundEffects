import numpy as np
import matplotlib.pyplot as plt

def plot_epochs(epochs, y, line):
    ep = np.arange(0, epochs)
    plt.plot(ep, y, line)
    plt.show()
