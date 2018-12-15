import tensorflow as tf
from networks import network_util
import plots


class Rnn(object):

    def __init__(self, num_inputs, num_classes, layers, dropout, epochs, l2_coef,
                 learning_rate, batch_size, save_model, save_path, max_seq_len):
        """
        Inits Rnn parameters
        :param num_inputs: size of input vector features
        :param num_classes: size of output vector classes
        :param layers: list of LSTM layer sizes
        :param dropout: keep probability for LSTM dropout
        :param epochs: num of training iterations
        :param l2_coef: L2 loss multiplier
        :param learning_rate: (prefered 0.001)
        :param batch_size: (prefered 8)
        :param save_model: if True then saves model to disk
        :param save_path: model save path
        :param max_seq_len: maximum length of input sequence
        """
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.l2_coef = l2_coef
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_model = save_model
        self.save_path = save_path
        self.max_seq_len = max_seq_len

        self.x = None
        self.y = None
        self.sequence_length = None
        self.dropout_keep_prob = None
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

    def _save(self, session, epoch):
        if self.save_model:
            self.saver.save(session, self.save_path, epoch)

    def _load(self, session, model_path):
        if model_path is not None:
            self.saver = tf.train.import_meta_graph(model_path)
            self.saver.restore(session, tf.train.latest_checkpoint("tmp/")) #pass config path?

    def train(self, train_data, train_labels, valid_data, valid_labels, plot_result=True):
        """
        Trains and validates model, return epoch and accuracy of best validation
        :param train_data:
        :param train_labels:
        :param valid_data:
        :param valid_labels:
        :param plot_result: if True then plots loss/accuracy
        :return: best epoch and its accuracy and result
        """
        best_acc = 0
        best_epoch = 0
        best_result = None
        loss_hist = []
        acc_hist = []
        best_top_5 = 0
        for epoch in range(self.epochs):
            accuracy = 0
            loss = 0
            i = 0
            for t_d, t_l in network_util.batch(train_data, train_labels, self.batch_size):
                feed_dict = {
                    self.x: t_d,
                    self.y: t_l,
                    self.dropout_keep_prob: self.dropout
                }
                _, train_step, train_predictions, t_loss, t_accuracy = self.session.run(
                    [self.optimizer, self.global_step, self.predictions, self.loss, self.accuracy], feed_dict
                )
                accuracy += t_accuracy
                loss += t_loss
                i += 1

            loss = loss / i
            accuracy = accuracy / i
            print("Step: {}, loss {}, acc {}".format(epoch + 1, loss, accuracy))
            if valid_data is not None and valid_labels is not None:
                feed_dict = {
                    self.x: valid_data,
                    self.y: valid_labels,
                    self.dropout_keep_prob: 1
                }
                _, result, test_loss, test_accuracy, top_5 = self.session.run(
                    [self.net, self.predictions, self.loss, self.accuracy, self.top_5], feed_dict
                )
                loss_hist.append((loss, test_loss))
                acc_hist.append(test_accuracy)
                print("Validation accuracy: {}, loss {}".format(test_accuracy, test_loss))
                if best_acc < test_accuracy:
                    best_acc = test_accuracy
                    best_epoch = epoch
                    best_result = result
                    best_top_5 = top_5
                    self._save(self.session, epoch)

                print("Best acc: {} at step {}, top 5 {}".format(best_acc, best_epoch + 1, best_top_5))
        if plot_result:
            plots.plot_epochs(self.epochs, loss_hist, ['r--', 'g--'])
            plots.plot_epochs(self.epochs, acc_hist, 'b--')
        return best_acc, best_epoch, best_result

    def test(self, test_data, test_labels, model_path):
        """
        Load model from model_path and tests data
        :param test_data:
        :param test_labels:
        :param model_path:
        :return result of testing and accuracy:
        """
        with tf.Session() as session:
            self._load(session, model_path)
            feed_dict = {
                self.x: test_data,
                self.y: test_labels,
                self.dropout_keep_prob: 1
            }
            _, result, test_accuracy = self.session.run(
                [self.net, self.predictions, self.accuracy], feed_dict
            )
            print("Test accuracy:", test_accuracy)
            print("Result:", result)
        return result, test_accuracy

    def build(self):
        """
        Builds neural network and starts session
        :return:
        """
        graph = tf.Graph()
        with graph.as_default():

            self.x = tf.placeholder(tf.float32,
                shape=[None, self.max_seq_len, self.num_inputs],
            )
            self.sequence_length = network_util.get_true_sequence_length(self.x, axis=1)
            self.y = tf.placeholder(tf.int64,
                shape=[None, self.num_classes],
            )

            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=())
            """Last layer weights and bias"""
            self.weight = {
                "out": tf.Variable(tf.truncated_normal(
                    [self.layers[-1], self.num_classes]
                ))
            }
            self.bias = {
                "out": tf.Variable(tf.zeros([self.num_classes]))
            }

            self.l2_reg = tf.constant(0.0) + \
                          tf.nn.l2_loss(self.weight["out"]) +\
                          tf.nn.l2_loss(self.bias["out"])

            def rnn(x, w, b, seq_len):
                rnn_cells = [tf.contrib.rnn.LSTMCell(layer) for layer in self.layers]
                rnn_cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in rnn_cells]
                rnn_cells = tf.contrib.rnn.MultiRNNCell(rnn_cells)
                out, _ = tf.nn.dynamic_rnn(rnn_cells, x,
                                           sequence_length=seq_len,
                                           dtype=tf.float32)
                out = network_util.transpose_to_last_values(out, seq_len)
                return tf.matmul(out, w["out"]) + b["out"]

            self.net = rnn(self.x, self.weight, self.bias, self.sequence_length)
            self.predictions = tf.argmax(self.net, axis=1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.net, labels=self.y
            )) + self.l2_coef * self.l2_reg
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predictions, tf.argmax(self.y, axis=1)), dtype=tf.float32)
            )
            self.top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.net, tf.argmax(self.y, axis=1), 5), dtype=tf.float32))
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)\
                .minimize(self.loss, global_step=self.global_step)
            #gvs, var = zip(*self.optimizer.compute_gradients(self.loss))
            #gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
            #self.optimizer = self.optimizer.apply_gradients(zip(gvs, var), global_step=self.global_step)
            self.saver = tf.train.Saver()
            #Windows GPU crash fix
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer())

    def close(self):
        """
        Call after end of calculation
        :return:
        """
        if self.session is not None:
            self.session.close()
            self.session = None





