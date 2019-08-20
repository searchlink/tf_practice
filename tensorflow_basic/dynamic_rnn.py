# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 20:27
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : dynamic_rnn.py
# @Software: PyCharm

'''lstm处理可变长序列'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
tf.logging.set_verbosity(tf.logging.INFO)


# toy data generator
class ToySequenceData():
    '''
     - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
     - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    '''
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3, max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []

        self.input_dict = {}

        for i in range(n_samples):
            seq_len = random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(seq_len)
            if random.random() < 0.5:
                rand_start = random.randint(0, max_value - seq_len)
                s = [[i] for i in range(rand_start, rand_start + seq_len)]
                s += [[0] for i in range(max_seq_len - seq_len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                s = [[random.randint(0, max_value)] for i in range(seq_len)]
                s += [[0] for i in range(max_seq_len - seq_len)]
                self.data.append(s)
                self.labels.append([0., 1.])

        self.input_dict["inputs_sequence"] = np.array(self.data, dtype=np.float32)
        self.input_dict["seq_len"] = np.array(self.seqlen, dtype=np.int32)
        self.labels = np.array(self.labels, dtype=np.float32)


# 生成训练数据和测试数据
trainset = ToySequenceData(n_samples=1000, max_seq_len=20)
testset = ToySequenceData(n_samples=500, max_seq_len=20)

# params
seq_max_len = 20
num_units = 64
n_classes = 2
batch_size = 128
learning_rate = 0.01
num_steps = 10000


# build model
def dynamic_rnn(input_dict):
    '''
    input shape: (batch_size, n_steps, n_input)
    [[[43], [653], [16], [353], [705], [139], [839], [198], [641], [772], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], ...]
    (batch_size, seq_max_len, 1)
    '''
    x = input_dict["inputs_sequence"]
    seq_len = input_dict["seq_len"]
    x = tf.unstack(x, seq_max_len, axis=1)  # (n_steps, batch_size, n_input) = > (seq_max_len, batch_size, 1)
    lstm_cell = rnn.BasicLSTMCell(num_units)
    # ValueError: If no initial_state is provided, dtype must be specified
    outputs, state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)   # 指定sequence_length参数进行可变长计算

    # 当执行动态计算时， 必须动态检索最后一个输出。 if a sequence length is 10, we need to retrieve the 10th output
    # build a custom op that for each sample in batch size, get its length and get the corresponding relevant output.
    # outputs： [seq_len, batch_size, num_units]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])  # [batch_size, seq_len, num_units]
    batch_size = tf.shape(outputs)[0]

    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seq_len - 1)   # last_index
    outputs = tf.gather(tf.reshape(outputs, [-1, num_units]), index)    # Defaults to the first dimension(根据索引分片取数据)
    outputs = tf.layers.dense(outputs, n_classes)  # [batch_size, n_classes]
    return outputs


def model_fn(features, labels, mode):
    logits = dynamic_rnn(features)

    pred_classes = tf.argmax(logits, axis=1)
    pred_probs = tf.nn.softmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={"accuracy": acc_op})

    return estim_specs


model = tf.estimator.Estimator(model_fn=model_fn)

# train
input_fn = tf.estimator.inputs.numpy_input_fn(x=trainset.input_dict, y=trainset.labels, batch_size=batch_size, num_epochs=None, shuffle=True)
model.train(input_fn, max_steps=num_steps)

# evaluate
input_fn = tf.estimator.inputs.numpy_input_fn(x=testset.input_dict, y=testset.labels, batch_size=batch_size, shuffle=True)
model.evaluate(input_fn)

# predict
n_samples = 4
test_d = ToySequenceData(n_samples=n_samples, max_seq_len=20)
input_fn = tf.estimator.inputs.numpy_input_fn(x=test_d.input_dict, shuffle=False)
preds = list(model.predict(input_fn))

for i in range(n_samples):
    print("test sample: ", test_d.input_dict["inputs_sequence"][i], "correspond sequence length is: ", test_d.input_dict["seq_len"][i])
    print("True image label is %d, prediction is %d" % (np.argmax(test_d.labels[i]), preds[i]))