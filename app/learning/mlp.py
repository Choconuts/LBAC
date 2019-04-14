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


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # no use
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.histogram_summary(layer_name + '/outputs', outputs)
    return outputs


# mlp全部全连接
vertex_num = 17436
learning_rate = 1e-4
keep_probability = 0.8 # 0.2 is good
batch_size = 128
model_path = '../data/model/shape/test1.ckpt'

# 数据输入占位符
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, 10], name="betas")
y_true = tf.placeholder(tf.float32, shape=[None, vertex_num * 3], name="displacements")

# 第1、2、3层: 全连接
full1 = full_conn_layer(x, 20)
full2 = full1 # full_conn_layer(full1, 500)
full3 = full_conn_layer(full2, vertex_num * 3, None)

# 损失优化、评估准确率
cross_entropy = tf.reduce_mean(tf.square(y_true - full3), 1)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.reduce_mean(tf.square(y_true - full3), 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_y = np.reshape(beta_gt.displacement[0:1], (-1, vertex_num * 3))
        for i in range(10000):
            batch = [beta_gt.betas[0:17], np.reshape(beta_gt.displacement[0:17], (-1, vertex_num * 3))]
            if i % 10 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1})
                print('step {}, training accuracy: {}'.format(i, train_accuracy))
            train_step.run(session=sess, feed_dict={x: batch[0], y_true: batch[1],  keep_prob: keep_probability})
        print('test accuracy: {}'.format(accuracy.eval(
            session=sess,
            feed_dict={x: beta_gt.betas[0:1], y_true: test_y, keep_prob: 1})))
        saver = tf.train.Saver()
        saver.save(sess, model_path)

    writer = tf.summary.FileWriter('logfile', tf.get_default_graph())
    writer.close()


# def test():
#     global disp
#     with tf.Session() as sess:
#         saver = tf.train.Saver()
#         saver.restore(sess, model_path)
#         disp = sess.run(full3, feed_dict={x: [[1.2, 0.3, 2.3, 1.2, -2.1, 0, 2.3, 0, 0, 0]], keep_prob: 1})
#         # disp = beta_gt.displacement[1]
#         np.reshape(beta_gt.displacement[5], (vertex_num, 3))


def predict(sess, betas):
    return sess.run(full3, feed_dict={x: [betas], keep_prob: 1})


disp = 0
# train()
# test()

if __name__ == '__main__':
    pass

# python /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorboard/tensorboard.py --logdir=logfile