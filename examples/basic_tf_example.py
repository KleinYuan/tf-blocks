import tensorflow as tf

data = [(-1, 1.1222), (-2, 2.2111)]
graph = tf.Graph()
with graph.as_default():
    x_feed = tf.placeholder(dtype=tf.float32, shape=2, name='input_1')
    y_feed = tf.placeholder(dtype=tf.float32, shape=2, name='input_2')
    cst = tf.Variable(0.2)
    res = tf.add(x_feed, y_feed, name='output_1')
    cst_r = tf.add(res, cst, name='output_2')

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    sess_res, sess_cst_r = sess.run([res, cst_r], feed_dict={
        x_feed: data[0],
        y_feed: data[1]
    })
    print sess_res
    print sess_cst_r
