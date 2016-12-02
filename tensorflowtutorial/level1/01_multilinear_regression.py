#!/usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf

data_x1 = np.linspace(-1, 1, 101)
data_x2 = np.linspace(-1, 1, 101)
data_x3 = np.linspace(-1, 1, 101)
data_y = 2 * data_x1 + 3 * data_x2 + 4 * data_x3 + np.random.randn(*data_x1.shape) * 0.2

X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
X3 = tf.placeholder("float")
Y = tf.placeholder("float")

w1 = tf.Variable(0.0, name="weight1")
w2 = tf.Variable(0.0, name="weight2")
w3 = tf.Variable(0.0, name="weight2")


def model(w1, w2, w3, x1, x2, x3):
    return tf.mul(w1, x1) + tf.mul(w2, x2), tf.mul(w3, x3)

output = model(w1, w2, w3, X1, X2, X3)

cost = tf.reduce_mean(tf.square(Y - output))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(100):
        sess.run(train, feed_dict={X1: data_x1, X2: data_x2, X3: data_x3, Y: data_y})
        W1 = sess.run(w1)
        W2 = sess.run(w2)
        W3 = sess.run(w3)
        print ("w1: %f; w2: %f; w3: %f" % (W1, W2, W3))








