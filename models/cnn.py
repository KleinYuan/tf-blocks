import tensorflow as tf
from models import blocks
from utils.logger import deep_log


class Net:

    @deep_log
    def __init__(self, dropout, num_output, num_conv, num_fc):
        print 'Initializing a net!'
        self.dropout = dropout
        self.num_output = num_output
        self.num_conv = num_conv
        self.num_fc = num_fc
        self.conv_kernel_config = {
            'size': 3,
            'depth_init': 32
        }
        self.pool_size = 2
        self.dropout_config = {
            'keep_prob_init': 0.9,
            'decay': 0.1
        }
        self.fc_relu_config = {
            'size': 1000
        }
        self.prediction = None

    def _construct_cnn_layers(self, input, training):
        conv_layers = []
        for i in range(0, self.num_conv):
            conv_input = input if len(conv_layers) == 0 else conv_layers[i - 1]
            dropout_param = self.dropout_config['keep_prob_init'] - float(i) * self.dropout_config['decay']

            with tf.variable_scope('conv%s' % i):
                print '\n[%s th] convolutional nn layer input shape is : %s' % (i + 1, conv_input.get_shape())
                conv = blocks.conv_relu(conv_input, kernel_size=self.conv_kernel_config['size'], depth=self.conv_kernel_config['depth_init']*i)
                pool = blocks.pool(conv, size=self.pool_size)
                pool = tf.cond(training, lambda: tf.nn.dropout(pool, keep_prob=dropout_param if self.dropout else 1.0), lambda: pool)
                conv_layers.append(pool)

        return conv_layers

    def _construct_fc_layers(self, flattened, training):
        fc_layers = []

        for j in range(0, self.num_fc):
            fc_input = flattened if len(fc_layers) == 0 else fc_layers[j - 1]
            dropout_param = self.dropout_config['keep_prob_init'] - (float(self.num_conv) + float(j)) * self.dropout_config['decay']

            with tf.variable_scope('fc%s' % (self.num_conv + j)):
                print '\n[%s th] fully connected layer input shape is : %s ' % (j + 1, fc_input.get_shape())
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
            prediction = blocks.fc(fc_layers[-1], size=self.num_output)

        return prediction

    def define_net(self, input, training):
        conv_layers = self._construct_cnn_layers(input=input, training=training)
        flattened = self._flattern_cnn_layers(layers=conv_layers)
        fc_layers = self._construct_fc_layers(flattened=flattened, training=training)
        self.prediction = self._construct_output_layer(fc_layers=fc_layers)


