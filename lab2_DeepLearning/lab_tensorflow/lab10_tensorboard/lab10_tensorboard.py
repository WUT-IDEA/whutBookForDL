# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import print_function

# dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../dataset/MNIST_data/", one_hot=True)

import tensorflow as tf

# hyper parameters
input_size = 784
hidden_size = 64
num_classes = 10

# placeholder
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('placehold'):
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='y')

    with tf.variable_scope('linear1'):
        # hidden layer
        weights = tf.Variable(tf.random_normal([input_size, hidden_size]), name='linear1_kernel')
        bias = tf.Variable(tf.random_normal([hidden_size]), name='linear1_bias')
        linear1 = tf.matmul(x, weights) + bias

        tf.summary.histogram("linear1_kernel", weights)
        tf.summary.histogram("linear1_bias", bias)
        tf.summary.histogram("linear1_output", linear1)

    with tf.name_scope('linear2'):
        linear2 = tf.layers.dense(inputs=linear1, units=num_classes, activation=tf.nn.softmax, name='linear2')

        tf.summary.histogram("linear2_output", linear2)

    y_ = linear2

    with tf.name_scope('loss_op'):
        loss_op = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), axis=-1))
        tf.summary.scalar("loss", loss_op)

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_op)

    with tf.name_scope('accuracy_op'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), dtype=tf.float32))
        tf.summary.scalar("loss", loss_op)

    summary = tf.summary.merge_all()

# initialize
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    # Create summary writer
    writer = tf.summary.FileWriter('history', sess.graph)
    writer.add_graph(sess.graph)
    global_step = 0

    # training
    for i in range(10):
        batch_size = 100
        batch_episodes = len(mnist.train.images) // batch_size
        for _ in xrange(batch_episodes):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {x: batch_xs, y: batch_ys}
            s, _ = sess.run([summary, train_op], feed_dict=feed_dict)
            writer.add_summary(s, global_step=global_step)
            global_step += 1

        # evaluate
        print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

'''
run "tensorboard --logdir history" in interminal
'''
