# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 14:29
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : kmeans.py
# @Software: PyCharm

'''
缺少交叉验证的代码
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
from tensorflow.examples.tutorials.mnist import input_data

# set log display
tf.logging.set_verbosity(tf.logging.WARN)

mnist = input_data.read_data_sets("./data", one_hot=True)

# params
num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 25 # The number of clusters
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels

# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# model
kmeans = KMeans(inputs=x, num_clusters=k, distance_metric="cosine", use_mini_batch=True)

# kmeans graph
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, training_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0]    # fix for cluster_idx being a tuple

# loss
avg_distance = tf.reduce_mean(scores)

# initialize
init_vars = tf.global_variables_initializer()

# Session
sess = tf.Session()
sess.run(init_vars)
sess.run(init_op, feed_dict={x: mnist.train.images})

# training
for i in range(num_steps + 1):
    # idx表示每个input对应的聚类中心id
    _, d, idx = sess.run([training_op, avg_distance, cluster_idx], feed_dict={x: mnist.train.images})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))
        print(idx)
        print(len(idx))
        print(set(idx))

# 计数每个聚类中心的label数
counts = np.zeros([k, num_classes])
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]

# 取每个聚类中心对应的最多标签
labels_map1 = np.argmax(counts, axis=1)
labels_map = [np.argmax(c) for c in counts]
print(labels_map1, labels_map)
print(type(labels_map1), type(labels_map))
labels_map = tf.convert_to_tensor(labels_map)
print(labels_map)
print(labels_map.shape)

# 评估op
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={x: test_x, y: test_y}))