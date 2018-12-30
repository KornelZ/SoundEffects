import tensorflow as tf
import numpy as np


def get_true_sequence_length(sequence, axis):
    """
    For LSTM: finds length of non zero elements in matrix of sequences
    :param sequence: matrix of sequences
    :param axis: orientation of sequences
    :return: length of sequence of non zero elements
    """
    correct = tf.sign(tf.abs(sequence))
    seq_len = tf.reduce_sum(correct, reduction_indices=axis)
    return tf.cast(seq_len, tf.int32)[:, 0]


def get_max_sequence_length(sequence_data):
    """
    Returns max length of sequence matrix
    :param sequence_data: sequence matrix
    :return: shape length of sequence matrix(including zero elements)
    """
    return sequence_data.shape[0]


def transpose_to_last_values(sequence, length):
    """
    Gathers elements at last non-zero position in LSTM
    :param sequence: sequence matrix
    :param length: length of sequence of non zero elements
    :return: vector of last non-zero elements of sequence matrix
    """
    _, max_seq_len, input_size = sequence.shape
    batch_size = tf.shape(sequence)[0]
    #seq = tf.transpose(sequence, [1, 0, 2])
    #seq = tf.gather(seq, length - 1)
    #sp = tf.shape(seq)
    #return tf.reshape(seq, [sp[1], sp[2]])
    index = tf.range(0, batch_size) * max_seq_len + (length - 1)
    return tf.gather(tf.reshape(sequence, [-1, input_size]), index)


def one_hot_encode(labels, num_classes):
    """
    Encodes labels vector to [0, 0, ..., 1, ... 0, 0] matrix
    :param labels: vect
    :param num_classes: number of distinct classes
    :return: returns matrix of size len(labels) x num_classes
    """
    x = np.asarray(labels)
    return np.eye(num_classes, dtype=np.float32)[x]


def batch(data, labels, batch_size):
    """
    Shuffles data and creates batch of size batch_size
    :param data: training data
    :param labels: training labels
    :param batch_size: size of returned batch
    :return: returns batch of training data and labels
    """
    perm = np.random.permutation(len(labels))
    data = data[perm, :]
    labels = labels[perm]
    length = len(labels)
    for i in range(0, length, batch_size):
        indices = range(i, i+batch_size)
        yield data.take(indices, mode="wrap", axis=0),\
              labels.take(indices, mode="wrap", axis=0)

