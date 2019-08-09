# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 17:28
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : gbdt.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
import tensorflow.contrib.boosted_trees.proto.learner_pb2 as gbdt_learner

# Set verbosity to display errors only (Remove this line for showing warnings)
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("./data", one_hot=False)

# params

learner_config = gbdt_learner.LearnerConfig()
learner_config.learn

gbdt_model = GradientBoostedDecisionTreeClassifier()
