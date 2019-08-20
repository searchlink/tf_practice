# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 13:36
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : mlp.py
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
## train params
batch_size = 128
num_epochs = 10
learning_rate = 0.01
num_steps = 1000    # Number of steps for which to train the model
## network params
n_hidden_1 = 256
n_hidden_2 = 256
num_classes = 10

# input function that would feed dict of numpy arrays into the model
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images},
                                              y=mnist.train.labels,
                                              batch_size=batch_size,
                                              num_epochs=num_epochs,
                                              shuffle=True)
def mlp(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

# define model(following TF Estimator Template)
def model_fn(features, labels, mode):
    logits = mlp(features)

    pred_classes = tf.argmax(logits, axis=1)
    # pred_probs = tf.nn.softmax(logits)

    # 实现训练、评估和预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # evaluate
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})
    return estim_specs

# build estimator
model = tf.estimator.Estimator(model_fn=model_fn)

# train
model.train(input_fn=input_fn, steps=num_steps)

# evaluate
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.test.images},
                                              y=mnist.test.labels,
                                              batch_size=batch_size,
                                              shuffle=True)
model.evaluate(input_fn)

# predict
n_images = 4
test_images = mnist.test.images[:n_images]
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': test_images}, shuffle=False)
preds = list(model.predict(input_fn))

for i in range(n_images):
    print("True image label is %d, prediction is %d" % (mnist.test.labels[i], preds[i]))



