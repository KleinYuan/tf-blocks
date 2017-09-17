import tensorflow as tf
from nolearn.lasagne import BatchIterator


def train(graph, epochs, batch_size, feed_dict, data, optimizer, loss_calculator, logdir, save_path, val_epoch=100, save_epoch=500, unittest=False):

    assert 'x' and 'y' and 'training' in feed_dict, 'feed_dict misses some key'
    assert 'train' and 'val' and 'test' in data, 'train or val or test datasets is missing in data'
    assert 'x' and 'y' in data['train'], 'x or y in data["train"] is missing'
    assert 'x' and 'y' in data['val'], 'x or y in data["val"] is missing'
    assert 'x' and 'y' in data['test'], 'x or y in data["test"] is missing'
    assert 'calculate' in dir(loss_calculator)

    x_feed_dict_key = feed_dict['x']
    y_feed_dict_key = feed_dict['y']
    is_training_key = feed_dict['training']
    x_train = data['train']['x']
    y_train = data['train']['y']
    x_val = data['val']['x']
    y_val = data['val']['y']
    x_test = data['test']['x']
    y_test = data['test']['y']

    if unittest: return True

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

        for epoch in range(epochs):
            batch_iterator = BatchIterator(batch_size=batch_size, shuffle=True)
            for x_train_batch, y_train_batch in batch_iterator(x_train, y_train):
                _, summary = list(sess.run([optimizer, summary_op], feed_dict={
                    x_feed_dict_key: x_train_batch,
                    y_feed_dict_key: y_train_batch,
                    is_training_key: True
                }))

            if epoch % val_epoch == 0:
                loss_train = loss_calculator.calculate(x_train, y_train)
                loss_val = loss_calculator.calculate(x_val, y_val)
                print '%s th epoch:\n'\
                    '   train loss: %s' \
                    '   val loss: %s'\
                    % (epoch, loss_train, loss_val)

            writer.add_summary(summary, epoch)

            if (epoch % save_epoch == 0) or (epoch == epochs - 1):
                snapshot_path = saver.save(sess=sess, save_path=save_path)
                print 'Snapshot of %s th epoch is saved to %s' % (epoch, snapshot_path)

                loss_test = loss_calculator.calculate(x_test, y_test)
                print '%s th epoch:\n' \
                      '   test loss: %s' \
                      % (epoch, loss_test)

