import tensorflow as tf
from config import config

class Graph:

    def __init__(self, input_shape, output_shape, net, learning_rate=0.001, momentum=0.9):
        print 'Initializing graph  ...'
        # e.g., input_shape = (200, 200, 3), output_shape=3
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.net = net

        self.input_placeholder_shape = tuple([None] + self.input_shape)
        self.output_placeholder_shape = tuple([None] + self.output_shape)
        self.graph = None
        self.init_graph = None

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.predictions = None
        self.loss = None
        self.optimizer = None

        self.x_feed = None
        self.y_feed = None
        self.is_training = None

        self._define_graph()

    def _define_graph(self):
        print 'Defining graph ...'
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_feed = tf.placeholder(tf.float32, shape=self.input_placeholder_shape, name=config.NODE_NAMES['inputs']['data'])
            self.y_feed = tf.placeholder(tf.float32, shape=self.output_placeholder_shape, name=config.NODE_NAMES['output'])
            self.is_training = tf.placeholder(tf.bool, name=config.NODE_NAMES['inputs']['bool'])

            self.predictions = self.net.define_net(self.x_feed, self.is_training)
            self.loss = tf.reduce_mean(tf.square(self.predictions - self.y_feed))
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("lr", self.learning_rate)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=self.momentum,
                                                        use_nesterov=True).minimize(self.loss)
            self.init_graph = tf.global_variables_initializer()

    def get_placeholders(self):
        return self.x_feed, self.y_feed, self.is_training

    def update_object_function(self):
        print 'Delegate for invoking customized loss and optimizer'

    def get_graph(self):
        return self.graph, self.init_graph