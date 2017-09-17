import tensorflow as tf


def fc(input, size):
    print 'fc input =  %s' % input.get_shape()
    weights = tf.get_variable('weights', shape=[input.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[size], initializer=tf.constant_initializer(0.0))
    return tf.matmul(input, weights) + biases


def conv_relu(input, kernel_size, depth):
    print 'conv_relu input =   %s' % input.get_shape()
    weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, input.get_shape()[3], depth],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable('biases', shape=[depth], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def pool(input, size):
    print 'pool input =  %s' % input.get_shape()
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def fc_relu(input, size):
    print 'fc_relu input =   %s' % input.get_shape()
    return tf.nn.relu(fc(input, size))
