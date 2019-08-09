# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 16:15
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : random_forest.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import resources
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensor_forest.python import tensor_forest

# set log display
tf.logging.set_verbosity(tf.logging.WARN)

mnist = input_data.read_data_sets("./data", one_hot=False)

# params
num_classes = 10
num_trees = 10
max_nodes = 1000
num_features = 784
num_steps = 500  # Total steps to train
batch_size = 1024  # The number of samples per batch

# placeholder
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int32, shape=[None])  # For random forest, labels must be integers (the class id)

# forest params
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes,
                                      num_features=num_features).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# get training graph and loss
train_op = forest_graph.training_graph(X, Y)
train_loss = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)    # (probabilities, tree_paths, variance)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

# sess
sess = tf.train.MonitoredSession()
sess.run(init_vars)

for i in range(1, num_steps + 1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, train_loss], feed_dict={X: batch_x, Y: batch_y})

    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))