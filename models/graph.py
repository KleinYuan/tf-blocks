import tensorflow as tf


class Graph:

    def __init__(self, input_shape, output_shape, net, learning_rate=0.001, momentum=0.9):
        # e.g., input_shape = (200, 200, 3), output_shape=3
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.net = net

        self.input_placeholder_shape = [None] + self.input_shape
        self.output_placeholder_shape = [None] + self.input_shape
        self.graph = tf.Graph()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.predictions = None
        self.loss = None
        self.optimizer = None

    def define_graph(self):
        with self.graph.as_default():
            self.x_feed = tf.placeholder(tf.float32, shape=self.input_placeholder_shape)
            self.y_feed = tf.placeholder(tf.float32, shape=self.output_placeholder_shape)
            self.is_training = tf.placeholder(tf.bool)
            self.predictions = self.net.define_net(self.x_feed, self.is_training)
            self.loss = tf.reduce_mean(tf.square(self.predictions - self.y_feed))
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("lr", self.learning_rate)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=self.momentum,
                                                        use_nesterov=True).minimize(self.loss)
