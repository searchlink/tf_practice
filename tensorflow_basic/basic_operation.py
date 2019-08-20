# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 10:29
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : basic_operation.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

# constant tensor operation
a = tf.constant(2.0, dtype=tf.float32)
b = tf.constant(3.0, dtype=tf.float32)
c = a + b

with tf.Session() as sess:
    print("a = %i, " % sess.run(a), "b = %i" % sess.run(b))
    print("c = %i" % sess.run(c))

# input placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = tf.add(a, b)
mul = tf.multiply(a, b)
c = a + b
with tf.Session() as sess:
    # TypeError: The value of a feed cannot be a tf.Tensor object(Python scalars, strings, lists, numpy ndarrays)
    print("add = %d" % sess.run(add, feed_dict={a: 2.0, b: 3.0}))
    print("c = %d" % sess.run(c, feed_dict={a: 2.0, b: 3.0}))
    print("mul = %d" % sess.run(mul, feed_dict={a: 2.0, b: 3.0}))

# matrix operation
matrix1 = tf.constant([[3., 3.]])  # (1, 2)
matrix2 = tf.constant([[3.], [3.]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result, result.shape)
    print(result[0], result[0].shape)

# string constant tensor
hello = tf.constant("Hello, tensorflow!")
sess = tf.Session()
print(sess.run(hello))