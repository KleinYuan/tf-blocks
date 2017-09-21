from nolearn.lasagne import BatchIterator


class Predictor:

    def __init__(self, sess, predict_graph, feed_dict, batch_size):
        print 'Initializing predictor ...'
        self.sess = sess
        self.predict_graph = predict_graph
        self.feed_dict = feed_dict
        self.batch_size = batch_size
        self.x_placeholder = self.feed_dict['x']
        self.is_training_placeholder = self.feed_dict['training']

    def predict_in_batch(self, x):
        print 'Evaluating predictions in batch with %s samples !' % len(x)
        predictions = []
        batch_iterator = BatchIterator(batch_size=self.batch_size)

        for x_feed, _ in batch_iterator(x):
            [y] = self.sess.run([self.predict_graph], feed_dict={
                self.x_placeholder: x_feed,
                self.is_training_placeholder: False
            })
            predictions.extend(y)
        return predictions
