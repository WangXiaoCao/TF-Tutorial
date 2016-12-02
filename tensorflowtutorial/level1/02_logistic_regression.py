#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# 获取初始权重的变量（随机产生的标准差为0.01的正态分布数据）
def get_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 创建模型：
def model(X, w):
    return tf.matmul(X, w)  # matmul是矩阵相乘

# 获取数据
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

print train_x.shape

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w = get_weight([784, 10])

output = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))

train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict = np.argmax(output, 1)
true = np.argmax(Y, 1)
accuracy = np.mean(predict == true)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(100):
        sess.run(train, feed_dict={X: train_x, Y: train_y})

        if i % 10 == 0:
            print (sess.run(cost, feed_dict={X: train_x, Y: train_y}))

    print sess.run(accuracy, {X: test_x, Y: test_y})

