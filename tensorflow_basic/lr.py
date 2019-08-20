# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 11:07
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : lr.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# set log display
tf.logging.set_verbosity(tf.logging.WARN)

# parameter
epochs = 1000
learning_rate = 0.01
display_step = 50

# Training Data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

n_samples = train_X.shape[0]

# placeholder input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# set model weights
W = tf.Variable(np.random.randn(), name="weights")
b = tf.Variable(np.random.randn(), name="bias")

# get prediction
pred = tf.add(tf.multiply(X, W), b)

# loss(mse)
loss = tf.reduce_sum(tf.pow(tf.subtract(pred, Y), 2)) / (2*n_samples)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # fit
    for epoch in range(epochs):
        for x, y in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if epoch % display_step == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
            print("Epoch #%4d: " % (epoch + 1), "cost = %.9f" % c, "W = %.5f" % sess.run(W), "b = %.5f" % sess.run(b))
    print("training is done!")
    last_loss = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print("Training cost = %.9f" % c, "W = %.5f" % sess.run(W), "b = %.5f" % sess.run(b))

    plt.plot(train_X, train_Y, "ro", label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), "-", label="Fitted line")
    plt.legend()
    plt.show()
