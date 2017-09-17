import unittest
import tensorflow as tf
from models.cnn import Net


class TestCNNModel(unittest.TestCase):

    def setUp(self):
        self.X = tf.placeholder(tf.float32, shape=(None, 125, 125, 3))
        self.training = tf.placeholder(tf.bool)

    def test_net(self):
        net = Net(dropout=True, num_output=10, num_conv=5, num_fc=4)
        net.define_net(input=self.X, training=self.training)
        self.assertIsNotNone(net.prediction)

if __name__ == '__main__':
    unittest.main()
