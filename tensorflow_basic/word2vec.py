# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 9:49
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : word2vec.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import urllib.request
import zipfile
from collections import Counter, deque
import random
import numpy as np
import tensorflow as tf

# Set verbosity to display errors only (Remove this line for showing warnings)
tf.logging.set_verbosity(tf.logging.ERROR)

# download data
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = 'text8.zip'
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print(filename)  # text8.zip
    print("Done!")
# Unzip the dataset file. Text has already been processed
with zipfile.ZipFile(data_path) as f:
    # tf.compat.as_str_any return string object
    text_words = tf.compat.as_str_any(f.read(f.namelist()[0])).lower().split()

# print(len(text_words))  # 17005207
print(text_words[:10])

# paras
## train paras
batch_size = 128
learning_rate = 0.1
num_steps = 100000 # 训练的步数
display_steps = 1000
eval_steps = 2000
model_path = './tmp/model.ckpt'  # 模型权重的保存地址
log_path = './tmp/logs/'
## word2vec paras
max_vocab_size = 50000
min_freq = 10
skip_window = 3  # How many words to consider left and right
num_skips = 2   # 从整个窗口中选取多少个不同的词作为我们的output word
embedding_size = 200
num_sampled = 64   # 采样出多少个负样本
## eval words
eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']
topk = 8

# build vocabulary(replace rare words with UNK token)
count = [('UNK', -1)]
count.extend(Counter(text_words).most_common(max_vocab_size - 1))
for i in range(len(count)-1, -1, -1):
    if count[i][1] < min_freq:
        count.pop(i)   # 逆序order
    else:
        break

vocab_size = len(count)
word2id = {}
for i, (word, _) in enumerate(count):
    word2id[word] = i
id2word = dict(zip(word2id.values(), word2id.keys()))
print([word2id[word] for word in eval_words])

# map source text data to id data
data = []
unk_count = 0
for word in text_words:
    # assign 0('UNK') if not in dict
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)

# update UNK counts
count[0] = ('UNK', unk_count)

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocab_size)
print("Most common words:", count[:10])

# generate sample for train and test
data_index = 0
def next_batch(batch_size, skip_window, num_skips):
    global data_index
    assert batch_size % num_skips == 0  # 确保每个batch包含一个词汇对应的所有样本
    assert num_skips <= 2 * skip_window
    # 定义多维数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    label = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # (words left and right + current one)
    buffer = deque(maxlen=span)

    if data_index + batch_size > len(data):
        data_index = 0  # 多次迭代
    buffer.extend(data[data_index:(data_index+span)])
    data_index += span  # 每次读入span长度的序列

    for i in range(batch_size // num_skips):
        # skip_window为中心词
        # context_words = [word for word in buffer if word != buffer[skip_window]]
        context_words = [w for w in range(span) if w != skip_window]
        num_skips_words = random.sample(context_words, num_skips)  # choose k unique random elements
        for j, context_word in enumerate(num_skips_words):
            batch[i * num_skips + j] = buffer[skip_window]
            label[i * num_skips + j, 0] = buffer[context_word]

        if data_index == len(data):
            # 重新开始
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, label

# test batch sample
batch, labels = next_batch(8, skip_window, num_skips)
for i in range(8):
    print(batch[i], id2word[batch[i]], "->", labels[i, 0], id2word[labels[i, 0]])

# placeholder
X = tf.placeholder(tf.int32, shape=[None], name='X')
Y = tf.placeholder(tf.int32, shape=[None, 1], name='Y')
eval_data = np.array([word2id[word] for word in eval_words])
valid_data = tf.constant(eval_data, dtype=tf.int32)

#
with tf.device("/cpu:0"):
    embedding = tf.Variable(tf.random_normal([vocab_size, embedding_size]), name="Embedding")
    X_embed = tf.nn.embedding_lookup(embedding, X)

    nce_weights = tf.Variable(tf.random_normal([vocab_size, embedding_size]), name='nce_weights')
    nce_bias = tf.Variable(tf.zeros([vocab_size]), name='nce_bias')

# nce loss
loss_op = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                        biases=nce_bias,
                                        labels=Y,
                                        inputs=X_embed,
                                        num_sampled=num_sampled,
                                        num_classes=vocab_size))

# 创建summary来monitor loss_op
tf.summary.scalar('loss', loss_op)

# if have other metrics to monitor, should Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)

# Evaluation(计算单词的嵌入向量与词汇表中所有向量的相似性)
norm = tf.square(tf.reduce_sum(tf.square(embedding), axis=1, keepdims=True))
norm_embed = embedding / norm
valid_embed_norm = tf.nn.embedding_lookup(norm_embed, valid_data)
similarity = tf.matmul(valid_embed_norm, norm_embed, transpose_b=True)

# initializer
init = tf.global_variables_initializer()

# Saver op to save and restore all the variables
saver = tf.train.Saver()

# train
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    total_loss = 0.
    for step in range(num_steps):
        batch_x, batch_y = next_batch(batch_size, skip_window, num_skips)
        _, loss, summary = sess.run([optimizer, loss_op, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})

        # Write logs at every iteration
        summary_writer.add_summary(summary, step)

        total_loss += loss

        if step % display_steps == 0:
            avg_loss = total_loss / display_steps
            print("Step #%d" % step, "Average loss=%.5f" % avg_loss)
            total_loss = 0.

        # Evaluate
        if step % eval_steps == 0:
            sim = similarity.eval()
            for i in range(len(eval_words)):
                # nearest = sim[i, :].argsort()[::-1][1:(topk+1)]
                # print("%s nearest neighbors: %s" % (eval_words[i], " ".join([id2word[j] for j in nearest])))
                nearest = (-sim[i, :]).argsort()[1:topk + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(topk):
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)

    # Save model weights to disk
    saver.save(sess, save_path=model_path)


# running a new session, 进行新的训练
with tf.Session() as sess:
    sess.run(init)

    # Restore model weights from previously saved model
    load_path = saver.restore(sess, save_path=model_path)

    # 重新训练
    total_loss = 0.
    for step in range(num_steps):
        batch_x, batch_y = next_batch(batch_size, skip_window, num_skips)
        _, loss = sess.run([optimizer, loss_op], feed_dict={X: batch_x, Y: batch_y})

        total_loss += loss

        if step % display_steps == 0:
            avg_loss = total_loss / display_steps
            print("Step #%d" % step, "Average loss=%.5f" % avg_loss)
            total_loss = 0.

        # Evaluate
        if step % eval_steps == 0:
            sim = similarity.eval()
            for i in range(len(eval_words)):
                # nearest = sim[i, :].argsort()[::-1][1:(topk+1)]
                # print("%s nearest neighbors: %s" % (eval_words[i], " ".join([id2word[j] for j in nearest])))
                nearest = (-sim[i, :]).argsort()[1:topk + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(topk):
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)

    print("再次迭代训练！")

# 启动tensorboard
"""
后台启动：
tensorboard --logdir=/tmp/pycharm_project_717/tf_practice/tmp/logs
浏览器打开：
http://192.168.15.27:6006
"""
