# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 9:15
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : mnist_data.py
# @Software: PyCharm

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(train_dir="./data", one_hot=True)

# load data
X_train = mnist.train.images
Y_train = mnist.train.labels

X_test = mnist.test.images
Y_test = mnist.test.labels

print(X_train.shape)

batch_X, batch_Y = mnist.train.next_batch(64)

print(mnist.train.epochs_completed)