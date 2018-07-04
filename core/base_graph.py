import tensorflow as tf


class BaseGraph:
    """
    A generic computation graph class, with only two constraints are
    1. The requirements for net instance passed in:
             - net should have method of define_net
             - net should have method of get_placeholders
    2. Losses can only be calculated with x_pl and y_pl (considering moving it out to network module)
    """
    graph = None
    init_graph = None

    losses = {}
    optimizers = {}

    predictions = None
    loss = None
    optimizer = None

    def __init__(self, net, config, logger):

        self.learning_rate = config.LEARNING_RATE
        self.momentum = config.MOMENTUM
        self.loss_type = config.LOSS_TYPE
        self.optimizer_type = config.OPTIMIZER_TYPE
        self.config = config
        self.logger = logger
        self.net = net

        self._define_graph()

    def _define_graph(self):
        self.logger.info('[Graph] Constructing graph now...')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.predictions = self.net.define_net()
            self.x_pl, self.y_pl, self.is_training_pl = self.net.get_placeholders()

            # Loss
            try:
                self.losses['mse'] = tf.reduce_mean(tf.square(self.predictions - self.y_pl))
                self.losses['cross_entropy'] = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_pl, logits=self.predictions))
            except ValueError as e:
                self.logger.warn(e)
            self.loss = self.losses[self.loss_type]

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("lr", self.learning_rate)

            # Optimizer
            self.optimizers['momentum'] = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                     momentum=self.momentum,
                                                                     use_nesterov=True).minimize(self.loss)
            self.optimizers['adam'] = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.optimizer = self.optimizers[self.optimizer_type]

            # Init nodes initializer
            self.init_graph = tf.global_variables_initializer()

        self.logger.info('[Graph] Graph constructed!')

    def get_placeholders(self):
        return self.x_pl, self.y_pl, self.is_training_pl

    def get_graph(self):
        return self.graph, self.init_graph

    def get_optimizer(self):
        return self.optimizer
