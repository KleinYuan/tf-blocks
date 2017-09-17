import tensorflow as tf
from nolearn.lasagne import BatchIterator
from services.predictServices import Predictor


def train(graph_model, epochs, batch_size, data, optimizer, loss_calculator, logdir, save_path, val_epoch=100, save_epoch=500, unittest=False):
    print 'Start training ...'

    assert 'train' and 'val' and 'test' in data, 'train or val or test datasets is missing in data'
    assert 'x' and 'y' in data['train'], 'x or y in data["train"] is missing'
    assert 'x' and 'y' in data['val'], 'x or y in data["val"] is missing'
    assert 'x' and 'y' in data['test'], 'x or y in data["test"] is missing'
    assert 'calculate' in dir(loss_calculator)

    print 'Asserting check passed!'

    x_train = data['train']['x']
    y_train = data['train']['y']
    x_val = data['val']['x']
    y_val = data['val']['y']
    x_test = data['test']['x']
    y_test = data['test']['y']

    graph = graph_model.graph
    x_placeholder, y_placeholder, is_training_placeholder = graph_model.get_placeholders()
    if unittest: return True

    print 'Running a session ...'
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

        predictor = Predictor(
            sess=sess,
            predict_graph=graph_model.predictions,
            feed_dict={
                    'x': x_placeholder,
                    'training': is_training_placeholder
                    },
            batch_size=batch_size)

        for epoch in range(epochs):
            print '%s th epoch, training ...' % epoch
            batch_iterator = BatchIterator(batch_size=batch_size, shuffle=True)
            for x_train_batch, y_train_batch in batch_iterator(x_train, y_train):
                _, summary = sess.run([optimizer, summary_op], feed_dict={
                    x_placeholder: x_train_batch,
                    y_placeholder: y_train_batch,
                    is_training_placeholder: True
                })

            if epoch % val_epoch == 0:
                predict_train = predictor.predict_in_batch(x_train)
                predict_val = predictor.predict_in_batch(x_val)

                loss_train = loss_calculator.calculate(predict_train, y_train)
                loss_val = loss_calculator.calculate(predict_val, y_val)
                print '%s th epoch:\n'\
                    '   train loss: %s' \
                    '   val loss: %s'\
                    % (epoch, loss_train, loss_val)

            writer.add_summary(summary, epoch)

            if (epoch % save_epoch == 0) or (epoch == epochs - 1):
                snapshot_path = saver.save(sess=sess, save_path=save_path)
                print 'Snapshot of %s th epoch is saved to %s' % (epoch, snapshot_path)

                predict_test = predictor.predict_in_batch(x_test)
                loss_test = loss_calculator.calculate(predict_test, y_test)
                print '%s th epoch:\n' \
                      '   test loss: %s' \
                      % (epoch, loss_test)

