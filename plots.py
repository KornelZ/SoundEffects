import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_epochs(epochs, y, line):
    ep = np.arange(0, epochs)
    plt.plot(ep, y, line)
    plt.show()

def confusion_matrix(pred, labels, num_classes):
    conf = np.zeros(shape=(num_classes, num_classes))
    for p, l in zip(pred.tolist(), labels):
        conf[p, l] += 1

    plt.imshow(conf, interpolation='nearest', cmap=plt.cm.gray_r)
    plt.title("Confusion Matrix")
    plt.colorbar()
    thresh = conf.max() / 2
    #for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
    #    plt.text(j, i, "{:,}".format(conf[i, j]),
    #             horizontalalignment="center",
    #             color="white" if conf[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Pred label')
    plt.xlabel('Exp label')
    plt.show()
    print(conf)
