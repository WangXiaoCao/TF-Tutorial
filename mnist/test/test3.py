#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV

np.random.seed(10)
x, y = make_moons(200, noise=0.20)

# plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)


def plot_decision_boundary(pred_func):
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)


# clf = LogisticRegressionCV()
# clf.fit(x, y)
#
# plot_decision_boundary(lambda x: clf.predict(x))
# plt.title("L")
# plt.show()

num_examples = len(x)
nn_input_dim = 2
nn_output_dim = 2

epsilon = 0.01
reg_lambda = 0.01


def calculate_loss(model):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_score = np.exp(z2)
    probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
    correct_log_probs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_log_probs)
    data_loss += reg_lambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return 1./num_examples * data_loss




