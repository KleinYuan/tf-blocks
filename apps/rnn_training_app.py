import tensorflow as tf
import numpy as np
from core.base_data_generator import BaseDataGenerator
from core.base_graph import BaseGraph
from core.base_trainer import BaseTrainer
from models.rnn_net import Net
from config import rnn_config as config
from sklearn.model_selection import train_test_split


class RNNDataGenerator(BaseDataGenerator):
	def load_data(self, *args, **kwargs):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

		x_train = np.expand_dims(x_train, axis=0)[0]
		y_train = np.expand_dims(y_train, axis=1)
		x_test = np.expand_dims(x_test, axis=0)[0]
		y_test = np.expand_dims(y_test, axis=1)

		# print("x shape: {}".format(x_train.shape))
		# print("y shape: {}".format(y_train.shape))

		self.train_data = {'x': x_train, 'y': y_train}

		x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=config.TEST_SIZE)

		self.test_data = {'x': x_test, 'y': y_test}
		self.val_data = {'x': x_val, 'y': y_val}


def main():
	# Config logger
	tf.logging.set_verbosity(tf.logging.INFO)
	logger = tf.logging
	# Initialize Four Modules: Data, Trainer, Net, Graph
	data_generator = RNNDataGenerator()
	net = Net(config=config, logger=logger)
	graph_model = BaseGraph(net=net, config=config, logger=logger)
	trainer = BaseTrainer(graph_model=graph_model, config=config, logger=logger)
	# Run Training
	trainer.train(data_generator=data_generator)


if __name__ == "__main__":
	main()
