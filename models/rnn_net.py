import tensorflow as tf
from core.base_net import BaseNet
from tensorflow.contrib import rnn


class Net(BaseNet):

    def __init__(self, config, logger):
        super(Net, self).__init__(config, logger)
        self.input_placeholder_shape = config.INPUT_SHAPE
        self.input_size = config.INPUT_SIZE
        self.node_names = config.NODE_NAMES
        self.num_hidden = config.NUM_HIDDEN
        self.num_classes = config.NUM_CLASSES

    def define_net(self, training=True):
        self.x_pl = tf.placeholder(tf.float32, shape=self.input_placeholder_shape, name=self.node_names['inputs']['data'])
        with tf.variable_scope('LSTM'):
            weights = tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
            biases = tf.Variable(tf.random_normal([self.num_classes]))

            _x = tf.unstack(self.x_pl, self.input_size, axis=1, name='unstack')
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

            predictions, states = rnn.static_rnn(lstm_cell, _x, dtype=tf.float32)
            predictions = tf.matmul(predictions[-1], weights) + biases
            self.y_pl = tf.placeholder(tf.float32, shape=predictions.shape, name=self.node_names['output'])
            self.is_training_pl = tf.placeholder(tf.bool, name=self.node_names['inputs']['bool'])

        return predictions
