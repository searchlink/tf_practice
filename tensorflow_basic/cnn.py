# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 15:24
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : cnn.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Set verbosity to display errors only (Remove this line for showing warnings)
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("./data", one_hot=False)

# params
num_input = 784
num_classes = 10
dropout = 0.25

learning_rate = 0.001
num_steps = 2000
batch_size = 128


def conv_net(x_dict, reuse, is_training):
    with tf.variable_scope("ConvNet", reuse=reuse):
        x = x_dict["images"]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, dropout, training=is_training)

        out = tf.layers.dense(fc1, num_classes)

    return out

def model_fn(features, labels, mode):
    logits_train = conv_net(features, reuse=False, is_training=True)
    logits_test = conv_net(features, reuse=True, is_training=False)

    # predict
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probs = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    #
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})
    return estim_specs

model = tf.estimator.Estimator(model_fn=model_fn)

# train
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images}, y=mnist.train.labels, batch_size=batch_size, num_epochs=None, shuffle=True)
model.train(input_fn, max_steps=num_steps)

# evaluate
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=batch_size, shuffle=True)
model.evaluate(input_fn)

# predict
n_images = 4
test_images = mnist.test.images[:n_images]
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': test_images}, shuffle=False)
preds = list(model.predict(input_fn))

for i in range(n_images):
    print("True image label is %d, prediction is %d" % (mnist.test.labels[i], preds[i]))