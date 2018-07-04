import tensorflow as tf


class BaseTrainer:
	session = None

	def __init__(self, graph_model, config, logger):
		self.graph_model = graph_model

		self.epochs = config.EPOCHS
		self.batch_size = config.BATCH_SIZE
		self.logdir = config.LOG_DIR
		self.save_path = config.SAVE_DIR
		self.val_epoch = config.VAL_EPOCH
		self.save_epoch = config.SAVE_EPOCH

		self.config = config
		self.logger = logger

	def eval(self, x, y, x_pl, y_pl, is_training_pl, tf_loss):
		loss = self.session.run([tf_loss], feed_dict={
			x_pl: x,
			y_pl: y,
			is_training_pl: False
		})
		return loss

	def _get_tf_config(self):
		tf_config = tf.ConfigProto(device_count=self.config.DEVICES)
		tf_config.gpu_options.allow_growth = True
		return tf_config

	def train(self, data_generator):
		self.logger.info('Start training ...')
		self.logger.info('Fetching graph and nodes of interests from graph model ...')
		graph, init_graph = self.graph_model.get_graph()
		optimizer = self.graph_model.get_optimizer()
		x_pl, y_pl, is_training_pl = self.graph_model.get_placeholders()

		self.logger.info('Opening a session and training started ...')
		with tf.Session(graph=graph, config=self._get_tf_config()) as self.session:

			self.session.run(init_graph)
			saver = tf.train.Saver()
			summary_op = tf.summary.merge_all()
			writer = tf.summary.FileWriter(logdir=self.logdir, graph=self.session.graph)

			for _epoch in range(self.epochs):
				self.logger.info('{} / {} th epoch, training ...'.format(_epoch, self.epochs))

				x_train_data, y_train_data = data_generator.train
				x_val_data, y_val_data = data_generator.val
				x_test_data, y_test_data = data_generator.test

				x_train_batch = None
				y_train_batch = None

				for x_train_batch, y_train_batch in data_generator.batch_iterator(x_train_data, y_train_data, batch_size=self.batch_size):

					_, summary = self.session.run([optimizer, summary_op], feed_dict={
						x_pl: x_train_batch,
						y_pl: y_train_batch,
						is_training_pl: True
					})

				if _epoch % self.val_epoch == 0:
					loss_train = self.eval(x=x_train_batch, y=y_train_batch, x_pl=x_pl, y_pl=y_pl,
					                       is_training_pl=is_training_pl, tf_loss=self.graph_model.loss)

					loss_val = self.eval(x=x_val_data, y=y_val_data, x_pl=x_pl, y_pl=y_pl,
					                     is_training_pl=is_training_pl, tf_loss=self.graph_model.loss)

					self.logger.info('{} th epoch:\ntrain loss: {}\nval loss: {}'.format(_epoch, loss_train, loss_val))

				writer.add_summary(summary, _epoch)

				if (_epoch % self.save_epoch == 0) or (_epoch == self.epochs - 1):
					snapshot_path = saver.save(sess=self.session, save_path="%seps_%s_" % (self.save_path, _epoch))
					self.logger.info('Snapshot of {} th epoch is saved to {}'.format(_epoch, snapshot_path))

					loss_test = self.eval(x=x_test_data,
					                      y=y_test_data,
					                      x_pl=x_pl,
					                      y_pl=y_pl,
					                      is_training_pl=is_training_pl,
					                      tf_loss=self.graph_model.loss)
					self.logger.info('{} th epoch:\ntest loss: {}'.format(_epoch, loss_test))
			save_path = saver.save(self.session, self.save_path)
			self.logger.info('Training ended and model file is in here: ', save_path)
