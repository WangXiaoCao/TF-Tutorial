#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

# 创建两个占位符，数据类型是float
a = tf.placeholder("float")
b = tf.placeholder("float")

# a, b两个值相乘:mul()
y = tf.mul(a, b)

# 创建tensorflow会话命名为sess.将两个数据喂给a,b，运行operation:y
with tf.Session() as sess:
    output1 = sess.run(y, feed_dict={a: 2, b: 4})
    output2 = sess.run(y, feed_dict={a: 10, b: 10})
    print output1
    print output2



