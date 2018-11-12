import rnn
import network_util
import numpy as np

def rnn_test():
    net = rnn.Rnn(
        num_inputs=2,
        num_classes=3,
        layers=[10],
        dropout=None,
        epochs=20,
        learning_rate=0.01,
        save_model=False,
        save_path=None
    )
    input = np.array([
        [[1, 2], [1, 2], [1, 4], [1, 5]],
        [[5, 5], [4, 6], [3, 8], [2, 9]],
        [[1, 3], [1, 3], [1, 4], [1, 4]],
        [[-5, 0], [-6, 0], [-6, 0], [0, 0]]
    ], dtype=np.float32)
    labels = network_util.one_hot_encode(np.array([0, 1, 0, 2]),
                                         num_classes=3)

    net.train(input, labels)

rnn_test()
