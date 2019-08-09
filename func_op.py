# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 10:03
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : func_op.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

# Set verbosity to display errors only (Remove this line for showing warnings)
tf.logging.set_verbosity(tf.logging.ERROR)

x = tf.constant([[1, 1, 1], [1, 1, 1]])

print(x/tf.reduce_sum(x, axis=1, keepdims=True))