import tensorflow as tf
from core.base_net import BaseNet


class Net(BaseNet):

    def __init__(self, config, logger):
        super(Net, self).__init__(config, logger)
        self.input_placeholder_shape = config.INPUT_SHAPE
        self.node_names = config.NODE_NAMES

    def define_net(self, training=True):
        self.logger.info('Defining a net ...')
        self.is_training_pl = tf.placeholder(tf.bool, name=self.node_names['inputs']['bool'])
        self.x_pl = tf.placeholder(tf.float32, shape=self.input_placeholder_shape, name=self.node_names['inputs']['data'])
        # TODO: Finish your architecture below and assign the results to predictions

        predictions = "Your network architecture results"
        self.y_pl = tf.placeholder(tf.float32, shape=predictions.shape, name=self.node_names['output'])

        return predictions
