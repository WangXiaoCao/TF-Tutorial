#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().decode("utf-8").replace("\n", "<eos>")


# a = tf.Variable([1, 2, 3])
# b = tf.reshape(a, [-1, 1])
#
# sess = tf.Session()
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
# print sess.run(a)
# print sess.run(b)

a = [[1, 2], [3, 4], [5, 6]]
b = a[-1][0]
print b