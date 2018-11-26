import tensorflow as tf
import numpy as np

def get_true_sequence_length(sequence, axis):
    correct = tf.sign(tf.abs(sequence))
    seq_len = tf.reduce_sum(correct, reduction_indices=axis)
    return tf.cast(seq_len, tf.int32)[:, 0]


def get_max_sequence_length(sequence_data):
    return sequence_data.shape[0]


def transpose_to_last_values(sequence, length):
    _, max_seq_len, input_size = sequence.shape
    batch_size = tf.shape(sequence)[0]
    index = tf.range(0, batch_size) * max_seq_len + (length - 1)
    return tf.gather(tf.reshape(sequence, [-1, input_size]), index)


def one_hot_encode(labels, num_classes):
    x = np.asarray(labels)
    return np.eye(num_classes, dtype=np.float32)[x]

def batch(data, labels, batch_size):
    perm = np.random.permutation(len(labels))
    data = data[perm, :]
    labels = labels[perm]
    for i in range(0, len(labels), batch_size):
        if i + batch_size >= len(labels):
            yield data[i:, :], labels[i:]
        else:
            yield data[i:i+batch_size, :], labels[i:i+batch_size]
