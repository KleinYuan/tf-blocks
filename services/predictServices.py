from nolearn.lasagne import BatchIterator


def predict_in_batch(x, sess, predict_graph, feed_dict, batch_size):
    print 'Evaluating predictions in batch with %s samples !' % len(x)
    predictions = []
    batch_iterator = BatchIterator(batch_size=batch_size)

    x_feed_dict_key = feed_dict['x']
    is_training_key = feed_dict['training']

    for x_feed, _ in batch_iterator(x):
        [y] = sess.run([predict_graph], feed_dict={
            x_feed_dict_key: x_feed,
            is_training_key: False
        })
        predictions.extend(y)
    return predictions
