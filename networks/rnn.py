import tensorflow as tf
from networks import network_util


class Rnn(object):

    def __init__(self, num_inputs, num_classes, layers, dropout, epochs,
                 learning_rate, batch_size, save_model, save_path, max_seq_len):
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_model = save_model
        self.save_path = save_path
        self.max_seq_len = max_seq_len

        self.x = None
        self.y = None
        self.sequence_length = None
        self.weight = None
        self.bias = None
        self.net = None
        self.predictions = None
        self.loss = None
        self.accuracy = None
        self.global_step = None
        self.optimizer = None
        self.session = None
        self.saver = None

    def _save(self, session):
        if self.save_model:
            self.saver.save(session, self.save_path, self.epochs)

    def _load(self, session, model_path):
        if self.save_path is not None:
            self.saver = tf.train.import_meta_graph(model_path)
            self.saver.restore(session, tf.train.latest_checkpoint("tmp/")) #pass config path?

    def train(self, train_data, train_labels):
        for epoch in range(self.epochs):
            feed_dict = {
                self.x: train_data,
                self.y: train_labels
            }
            _, train_step, train_predictions, train_loss, train_accuracy = self.session.run(
                [self.optimizer, self.global_step, self.predictions, self.loss, self.accuracy], feed_dict
            )
            print("Step: {}, loss {}, acc {}".format(train_step, train_loss, train_accuracy))

    def test(self, test_data, test_labels, model_path):
        with tf.Session() as session:
            self._load(session, model_path)
            feed_dict = {
                self.x: test_data,
                self.y: test_labels
            }
            _, result, test_accuracy = self.session.run(
                [self.net, self.predictions, self.accuracy], feed_dict
            )
            print("Test accuracy:", test_accuracy)
            print("Result\n {} {}".format(_, result))
        return result, test_accuracy

    def build(self):
        graph = tf.Graph()
        with graph.as_default():
            self.x = tf.placeholder(tf.float32,
                shape=[None, self.max_seq_len, self.num_inputs],
            )
            self.sequence_length = network_util.get_true_sequence_length(self.x, axis=1)
            self.y = tf.placeholder(tf.float32,
                shape=[None, self.num_classes],
            )

            self.weight = {
                "out": tf.Variable(tf.truncated_normal(
                    [self.layers[-1], self.num_classes]
                ))
            }
            self.bias = {
                "out": tf.Variable(tf.zeros([self.num_classes]))
            }

            def rnn(x, w, b, seq_len):
                rnn_cells = [tf.contrib.rnn.BasicRNNCell(layer) for layer in self.layers]
                rnn_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells)
                out, _ = tf.nn.dynamic_rnn(rnn_cells, x,
                                           sequence_length=seq_len,
                                           dtype=tf.float32)
                out = network_util.transpose_to_last_values(out, seq_len)
                return tf.nn.softmax(tf.matmul(out, w["out"]) + b["out"])

            self.net = rnn(self.x, self.weight, self.bias, self.sequence_length)
            self.predictions = tf.argmax(self.net, axis=1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.net, labels=self.y
            ))
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predictions, tf.argmax(self.y, axis=1)), dtype=tf.float32)
            )
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)\
                .minimize(self.loss, global_step=self.global_step)
            self.saver = tf.train.Saver()
            self.session = tf.Session()
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer())
                self._save(self.session)





