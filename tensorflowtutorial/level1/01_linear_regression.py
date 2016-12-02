#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

# 创造一组x,y的数据，y是x 线性相关，并加入一点点噪声
data_x = np.linspace(-1, 1, 101)
data_y = 3 * data_x + np.random.randn(*data_x.shape) * 0.33


# 设置占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")


# 创建模型
def model(x, w):
    return tf.mul(x, w)


w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
y_model = model(X, w)

# 损失函数：平方误差
cost = tf.square(Y - y_model)

# 使用梯度下降法训练模型，估计出最优参数，使得损失最小
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 创建一个tensorflow会话
with tf.Session() as sess:
    # 初始化所有变量
    tf.initialize_all_variables().run()

    # 训练10次
    for i in range(10):
        for (a, b) in zip(data_x, data_y):
            sess.run(train, feed_dict={X: a, Y: b})
        if i % 1 == 0:
            print(sess.run(w))
    print(sess.run(w))







