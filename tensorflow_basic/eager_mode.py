# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 10:28
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : eager_mode.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

# set log display
tf.logging.set_verbosity(tf.logging.WARN)
# set eager api(program startup)
tf.enable_eager_execution()

a = tf.constant(2)
print(a)
print("a=%i" % a)
print("a=%d" % a)

b = tf.constant(3)

c = a + b
print("a + b = %i" % c)

# tf tensor
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
# numpy array
b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

c = a + b
print("Tensor: \n c = %s" % c)

d = tf.matmul(a, b)
print("Tensor: \n d = %s" % d)

print(a.shape)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print("a[%d][%d] = %d" % (i, j, a[i][j]))