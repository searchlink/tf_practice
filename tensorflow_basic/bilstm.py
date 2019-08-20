# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 20:20
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : bilstm.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# Set verbosity to display errors only (Remove this line for showing warnings)
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("./data", one_hot=True)

# params
batch_size = 128
learning_rate = 0.001
timesteps = 28
num_input = 28
num_units = 128
num_classes = 10
num_steps = 10000

def bilstm_net(x_dict):
    x = x_dict["images"]
    x = tf.reshape(x, [-1, 28, 28])
    x = tf.unstack(x, 28, axis=1)   # list 28 * (64, 28)
    lstm_fw_cell = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    # outputs is a length `T` list of outputs (one for each input), which are depth-concatenated forward and backward outputs.
    # output_state_fw is the final state of the forward rnn.
    # output_state_bw is the final state of the backward rnn.
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    out = tf.layers.dense(outputs[-1], num_classes)
    return out

def model_fn(features, labels, mode):
    logits = bilstm_net(features)

    pred_classes = tf.argmax(logits, axis=1)
    pred_probs = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, pred_classes)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(mode,
                                             predictions=pred_classes,
                                             loss=loss_op,
                                             train_op=train_op,
                                             eval_metric_ops={"accuracy": acc_op})
    return estim_specs

model = tf.estimator.Estimator(model_fn)

# train
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images},
                                              y=mnist.train.labels,
                                              batch_size=batch_size,
                                              num_epochs=None,
                                              shuffle=True)
model.train(input_fn, max_steps=num_steps)

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
    print("True image label is %d, prediction is %d" % (np.argmax(mnist.test.labels[i]), preds[i]))