# coding=utf-8
import os
import sys
import tensorflow as tf
import numpy as np
import time
from app.learning.ground_truths import beta_gt


def conv_pool_layer(input_layer, output_size, kernel=5):
    shape = [kernel, kernel, int(np.shape(input_layer)[3]), output_size]
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))     # 权值
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))      # 偏置
    conv = tf.nn.conv2d(input=input_layer, filter=w, strides=[1, 1, 1, 1], padding='SAME')  # 卷积
    p = tf.nn.max_pool(tf.nn.relu(conv + b), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return p    # 池化


def full_conn_layer(input_layer, output_size, activate=tf.nn.relu):
    dim = 1
    input_shape = np.shape(input_layer)
    for i in range(1, len(input_shape)):
        dim *= int(input_shape[i])      # 拉成向量
    if len(input_shape) > 2:
        input_layer = tf.reshape(input_layer, [-1, dim])
    shape = [dim, output_size]
    w_fc = tf.Variable(tf.truncated_normal(shape, stddev=0.01))  # 权值
    b_fc = tf.Variable(tf.constant(0.01, shape=[output_size]))   # 偏置
    h_fc = activate(tf.matmul(input_layer, w_fc) + b_fc)        # 激活函数
    return h_fc


# mlp全部全连接

# 数据输入占位符
x = tf.placeholder('float', shape=[None, 10])
y_true = tf.placeholder('float', shape=[None, 17436 * 3])

# 第1、2、3层: 全连接
full1 = full_conn_layer(x, 20)
full2 = full1 # full_conn_layer(full1, 500)
full3 = full_conn_layer(full2, 17436 * 3)

# 损失优化、评估准确率
cross_entropy = tf.reduce_mean(tf.square(y_true - full3), 1)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.reduce_mean(tf.square(y_true - full3), 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_y = np.reshape(beta_gt.displacement[0:1], (-1, 17436 * 3))
    for i in range(1000):
        batch = [beta_gt.betas[1:17], np.reshape(beta_gt.displacement[1:17], (-1, 17436 * 3))]
        if i % 10 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_true: batch[1]})
            print('step {}, training accuracy: {}'.format(i, train_accuracy))
        train_step.run(session=sess, feed_dict={x: batch[0], y_true: batch[1]})
    print('test accuracy: {}'.format(accuracy.eval(session=sess, feed_dict={x: beta_gt.betas[0:1], y_true: test_y})))
    saver = tf.train.Saver()
    saver.save(sess, '../test/test1.ckpt')
    disp = sess.run(full3, feed_dict={x: beta_gt.betas[5:6], y_true: test_y})
    # disp = beta_gt.displacement[1]
    np.reshape(beta_gt.displacement[5], (17436, 3))


writer = tf.summary.FileWriter('logfile', tf.get_default_graph())
writer.close()

if __name__ == '__main__':
    pass

# python /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorboard/tensorboard.py --logdir=logfile