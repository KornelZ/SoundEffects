import numpy as np
import tensorflow as tf
import network_util

class Rnn(object):

    def __init__(self, num_inputs, num_classes, layers, dropout, epochs,
                 learning_rate, save_model, save_path):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        #self.batch_size = batch_size
        self.save_model = save_model
        self.save_path = save_path

    def save(self):
        pass

    def test(self):
        pass

    def train(self, train_data, train_labels):
        graph = tf.Graph()
        with graph.as_default():
            max_len = network_util.get_max_sequence_length(train_data)

            x = tf.placeholder(tf.float32,
                shape=[None, max_len, self.num_inputs],
            )
            sequence_len = network_util.get_true_sequence_length(x, axis=1)
            y = tf.placeholder(tf.float32,
                shape=[None, self.num_classes],
            )
            #dropout_keep_probability = tf.placeholder(tf.float32)

            weight = {
                "out": tf.Variable(tf.truncated_normal(
                    [self.layers[0], self.num_classes]
                ))
            }
            bias = {
                "out": tf.Variable(tf.zeros([self.num_classes]))
            }
            def rnn(x, w, b, seq_len):
                rnn_cell = tf.contrib.rnn.BasicRNNCell(self.layers[0])
                out, _ = tf.nn.dynamic_rnn(rnn_cell, x,
                                           sequence_length=sequence_len,
                                           dtype=tf.float32)
                out = network_util.transpose_to_last_values(out, seq_len)
                return tf.nn.softmax(tf.matmul(out, w["out"]) + b["out"])

            logits = rnn(x, weight, bias, sequence_len)
            pred = tf.argmax(logits, axis=1)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=y
            ))
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(pred, tf.argmax(y, axis=1)), dtype=tf.float32)
            )
            step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)\
                .minimize(loss, global_step=step)

            session = tf.Session()
            with session.as_default():
                session.run(tf.global_variables_initializer())

                for epoch in range(self.epochs):
                    feed_dict = {
                        x: train_data,
                        y: train_labels
                    }
                    _, c_step, c_loss, c_accuracy = session.run(
                        [optimizer, step, loss, accuracy], feed_dict
                    )
                    print("Epoch {}: step {}, loss {}, acc {}".format(epoch, c_step, c_loss, c_accuracy))





