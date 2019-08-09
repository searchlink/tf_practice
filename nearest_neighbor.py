# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 15:51
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : nearest_neighbor.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set log display
tf.logging.set_verbosity(tf.logging.WARN)

mnist = input_data.read_data_sets("./data", one_hot=True)

x_train, y_train = mnist.train.next_batch(5000)
x_test, y_test = mnist.test.next_batch(200)

# inputs
xtr = tf.placeholder(tf.float32, [None, 784])
xte = tf.placeholder(tf.float32, [784])

# distant 计算L1范数
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), axis=1)
# predict
predict = tf.argmin(distance)

# initialize
init = tf.global_variables_initializer()

# train
with tf.Session() as sess:
    sess.run(init)

    tp = 0
    for i in range(len(x_test)):
        nn_index = sess.run(predict, feed_dict={xtr: x_train, xte: x_test[i, :]})

        print("Test", i, "Prediction:", np.argmax(y_train[nn_index]), "True Class:", np.argmax(y_test[i]))

        if np.argmax(y_train[nn_index]) == np.argmax(y_test[i]):
            tp += 1

    accuracy = tp / len(x_test)
    print("done!")
    print("Accuracy:", accuracy)

