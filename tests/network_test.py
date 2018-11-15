from networks import network_util, rnn
import numpy as np

def rnn_test():
    net = rnn.Rnn(
        num_inputs=2,
        num_classes=3,
        layers=[20, 10],
        dropout=None,
        epochs=100,
        learning_rate=0.001,
        batch_size=4,
        save_model=True,
        save_path="tmp/model.tf",
        max_seq_len=4
    )
    input = np.array([
        [[1, 2], [1, 2], [1, 4], [1, 5]],
        [[5, 5], [4, 6], [3, 8], [2, 9]],
        [[1, 3], [1, 3], [1, 4], [1, 4]],
        [[-5, 0], [-6, 0], [-6, 0], [0, 0]]
    ], dtype=np.float32)
    labels = network_util.one_hot_encode(np.array([0, 1, 0, 2]),
                                         num_classes=3)
    net.build()
    net.train(input, labels)
    test_input = np.array([[[-5, 0], [-6, 0], [-5, 0], [0, 0]]],
                          dtype=np.float32)
    labels = network_util.one_hot_encode(np.array([2]), num_classes=3)
    net.test(test_input, labels, "tmp/model.tf-100.meta")


rnn_test()
