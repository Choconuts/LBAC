# coding=utf-8
import os
import sys
import tensorflow as tf
import numpy as np
import time
from app.learning.ground_truths import beta_gt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    d_fc = tf.matmul(input_layer, w_fc) + b_fc
    d_fc = tf.nn.dropout(d_fc, keep_prob)
    if activate != None:
        h_fc = activate(d_fc)        # 激活函数
    else:
        h_fc = d_fc
    h_fc = tf.nn.dropout(h_fc, keep_prob)
    return h_fc


# mlp全部全连接
vertex_num = 7366 #17436
learning_rate = 1e-4
keep_probability = 0.99 # 0.2 is good
batch_size = 10
model_path = '../data/model/shape/test2.ckpt'


def shape_graph():
    global keep_prob, x_shape_input, y_shape_true, shape_final_full, shape_cross_entropy, shape_train_step, shape_accuracy

    # 数据输入占位符
    keep_prob = tf.placeholder(tf.float32)
    x_shape_input = tf.placeholder(tf.float32, shape=[None, 10], name="betas")
    y_shape_true = tf.placeholder(tf.float32, shape=[None, vertex_num * 3], name="displacements")

    # 第1、2、3层: 全连接
    full1 = full_conn_layer(x_shape_input, 20)
    full2 = full1 # full_conn_layer(full1, 500)
    shape_final_full = full_conn_layer(full2, vertex_num * 3, None)

    # 损失优化、评估准确率
    shape_cross_entropy = tf.reduce_mean(tf.square(y_shape_true - shape_final_full))
    shape_train_step = tf.train.AdamOptimizer(learning_rate).minimize(shape_cross_entropy)
    correct_prediction = tf.reduce_mean(tf.square(y_shape_true - shape_final_full), 1)
    shape_accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


def train():

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_y = np.reshape(beta_gt.displacement[5:6], (-1, vertex_num * 3)) * 100
        for i in range(5000):
            batch = {}
            batch[0] = np.tile(beta_gt.betas[5], (batch_size, 1))
            batch[1] = np.tile(np.reshape(beta_gt.displacement[5:6], (-1, vertex_num * 3))[0], (batch_size, 1))
            batch = [beta_gt.betas[0:17], np.reshape(beta_gt.displacement[0:17], (-1, vertex_num * 3))]
            if i % 10 == 0:
                train_accuracy = shape_accuracy.eval(session=sess, feed_dict={x_shape_input: batch[0], y_shape_true: batch[1], keep_prob: 1})
                print('step {}, training accuracy: {}'.format(i, train_accuracy))

            shape_train_step.run(session=sess, feed_dict={x_shape_input: batch[0], y_shape_true: batch[1], keep_prob: keep_probability})
        print('test accuracy: {}'.format(shape_accuracy.eval(
            session=sess,
            feed_dict={x_shape_input: beta_gt.betas[5:6], y_shape_true: test_y, keep_prob: 1})))
        saver = tf.train.Saver()
        saver.save(sess, model_path)

    writer = tf.summary.FileWriter('logfile', tf.get_default_graph())
    writer.close()


def predict(sess, betas):
    return sess.run(shape_final_full, feed_dict={x_shape_input: [betas], keep_prob: 1})


if __name__ == '__main__':
    shape_graph()
    train()

# python /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorboard/tensorboard.py --logdir=logfile