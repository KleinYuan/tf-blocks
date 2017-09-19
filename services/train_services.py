import tensorflow as tf
from nolearn.lasagne import BatchIterator
from services.predict_services import Predictor


class Trainer:

    def __init__(self, graph_model, epochs, batch_size, logdir, save_path, val_epoch=100, save_epoch=500):
        self.graph_model = graph_model
        self.epochs = epochs
        self.val_epoch = val_epoch
        self.save_epoch = save_epoch
        self.batch_size = batch_size
        self.logdir = logdir
        self.save_path = save_path
        self.session = None

    def train(self, data, loss_calculator):
        print 'Start training ...'

        x_train = data['train']['x']
        y_train = data['train']['y']
        x_val = data['val']['x']
        y_val = data['val']['y']
        x_test = data['test']['x']
        y_test = data['test']['y']

        graph, init_graph = self.graph_model.get_graph()
        optimizer = self.graph_model.optimizer
        x_placeholder, y_placeholder, is_training_placeholder = self.graph_model.get_placeholders()

        print 'Running a session ...'
        tf_config = tf.ConfigProto(device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=tf_config) as self.session:

            self.session.run(init_graph)
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logdir=self.logdir, graph=self.session.graph)

            predictor = Predictor(
                sess=self.session,
                predict_graph=self.graph_model.predictions,
                feed_dict={
                        'x': x_placeholder,
                        'training': is_training_placeholder
                        },
                batch_size=self.batch_size)

            for epoch in range(self.epochs):
                print '%s / %s th epoch, training ...' % (epoch, self.epochs)
                batch_iterator = BatchIterator(batch_size=self.batch_size, shuffle=True)
                for x_train_batch, y_train_batch in batch_iterator(x_train, y_train):

                    _, summary = self.session.run([optimizer, summary_op], feed_dict={
                        x_placeholder: x_train_batch,
                        y_placeholder: y_train_batch,
                        is_training_placeholder: True
                    })

                if epoch % self.val_epoch == 0:
                    print '[Validating Round]'
                    predict_train = predictor.predict_in_batch(x_train)
                    predict_val = predictor.predict_in_batch(x_val)

                    loss_train = loss_calculator.calculate(predict_train, y_train)
                    loss_val = loss_calculator.calculate(predict_val, y_val)
                    print '%s th epoch:\n'\
                        '   train loss: %s' \
                        '   val loss: %s'\
                        % (epoch, loss_train, loss_val)

                writer.add_summary(summary, epoch)

                if (epoch % self.save_epoch == 0) or (epoch == self.epochs - 1):
                    print '[Testing Round]'
                    snapshot_path = saver.save(sess=self.session, save_path=self.save_path)
                    print 'Snapshot of %s th epoch is saved to %s' % (epoch, snapshot_path)

                    predict_test = predictor.predict_in_batch(x_test)
                    loss_test = loss_calculator.calculate(predict_test, y_test)
                    print '%s th epoch:\n' \
                          '   test loss: %s' \
                          % (epoch, loss_test)
