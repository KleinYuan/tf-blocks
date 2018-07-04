import tensorflow as tf

from core.base_net import BaseNet
from utils import blocks


class Net(BaseNet):

    def __init__(self, config, logger):
        super(Net, self).__init__(config, logger)
        self.dropout = config.DROPOUT
        self.num_features = config.NUM_FEATURES
        self.num_conv = config.NUM_CONV
        self.num_fc = config.NUM_FC
        self.input_placeholder_shape = config.INPUT_SHAPE
        self.node_names = config.NODE_NAMES

        self.cnn_config = config.CNN_CONFIG
        self.conv_kernel_config = self.cnn_config['conv_kernel_config']
        self.pool_size = self.cnn_config['pool_size']
        self.dropout_config = self.cnn_config['dropout_config']
        self.fc_relu_config = self.cnn_config['fc_relu_config']

    def _construct_cnn_layers(self, input, training):
        conv_layers = [input]
        for i in range(0, self.num_conv):
            conv_input = conv_layers[-1]
            dropout_param = self.dropout_config['keep_prob_init'] - float(i) * self.dropout_config['decay']
            kernel_size = self.conv_kernel_config['size']
            depth = self.conv_kernel_config['depth_init']*(1+i)
            with tf.variable_scope('conv%s' % (i + 1)):
                self.logger.info('\n[%s th] CNN layer input shape is : %s' % (i + 1, conv_input.get_shape()))
                conv = blocks.conv_relu(conv_input, kernel_size=kernel_size, depth=depth)
                pool = blocks.pool(conv, size=self.pool_size)
                pool = tf.cond(training, lambda: tf.nn.dropout(pool, keep_prob=dropout_param if self.dropout else 1.0), lambda: pool)
                conv_layers.append(pool)

        return conv_layers

    def _construct_fc_layers(self, flattened, training):
        fc_layers = [flattened]
        for j in range(0, self.num_fc):
            fc_input = fc_layers[-1]
            dropout_param = self.dropout_config['keep_prob_init'] - (float(self.num_conv) + float(j)) * self.dropout_config['decay']

            with tf.variable_scope('fc%s' % (self.num_conv + j + 1)):
                self.logger.info('\n[%s th] FC layer input shape is : %s ' % (self.num_conv + j + 1, fc_input.get_shape()))
                fc = blocks.fc_relu(fc_input, size=self.fc_relu_config['size'])
                if self.num_fc - j > 1:
                    fc = tf.cond(training, lambda: tf.nn.dropout(fc, keep_prob=dropout_param if self.dropout else 1.0), lambda: fc)
                fc_layers.append(fc)

        return fc_layers

    @staticmethod
    def _flattern_cnn_layers(layers):
        last_layer = layers[-1]
        shape = last_layer.get_shape().as_list()
        flattened = tf.reshape(last_layer, [-1, shape[1] * shape[2] * shape[3]])

        return flattened

    def _construct_output_layer(self, fc_layers):
        with tf.variable_scope('out'):
            prediction = blocks.fc(fc_layers[-1], size=self.num_features)

        return prediction

    def define_net(self, training=True):
        self.logger.info('Defining a net ...')
        self.is_training_pl = tf.placeholder(tf.bool, name=self.node_names['inputs']['bool'])
        self.x_pl = tf.placeholder(tf.float32, shape=self.input_placeholder_shape, name=self.node_names['inputs']['data'])

        conv_layers = self._construct_cnn_layers(input=self.x_pl, training=self.is_training_pl)
        flattened = self._flattern_cnn_layers(layers=conv_layers)
        fc_layers = self._construct_fc_layers(flattened=flattened, training=self.is_training_pl)
        predictions = self._construct_output_layer(fc_layers=fc_layers)
        self.y_pl = tf.placeholder(tf.float32, shape=predictions.shape, name=self.node_names['output'])

        return predictions

    def get_placeholders(self):
        return self.x_pl, self.y_pl, self.is_training_pl
