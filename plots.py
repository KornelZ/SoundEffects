import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_epochs(epochs, y, line):
    """
    Plots one or multiple line plots across epochs
    :param epochs: x axis epochs
    :param y: y axis values may be scalar or list
    :param line: line type for each y
    :return:
    """
    ep = np.arange(0, epochs)
    if hasattr(y[0], '__len__'):
        for i in range(len(y[0])):
            plt.plot(ep, [val[i] for val in y], line[i])
    else:
        plt.plot(ep, y, line)
    plt.show()

def confusion_matrix(pred, labels, num_classes):
    """
    Draws confusion matrix
    :param pred: vector of predictions
    :param labels: vector of expected labels
    :param num_classes: number of distinct classes
    :return:
    """
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
