# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 13:39
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : logistic_regression.py
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

# params
learning_rate = 0.01
epochs = 20
batch_size = 64
display_epochs = 1

# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# get prediction
pred = tf.nn.softmax(tf.matmul(x, W) + b)   # softmax

# loss
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# initalizer
init = tf.global_variables_initializer()

# train and output
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        total_loss = 0.
        steps = mnist.train.num_examples // batch_size

        for step in range(steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 需要避免redefine variable
            _, batch_loss = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

            total_loss = batch_loss + total_loss
        avg_loss = total_loss / steps

        if (epoch + 1) % display_epochs == 0:
            print("Epoch #%4d: " % (epoch + 1), "loss = %.9f" % avg_loss)

    print("training done!")

    # 预测评估
    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: %.5f" % (sess.run(accuracy, feed_dict={x: mnist.test.images[:3000], y: mnist.test.labels[:3000]})))



