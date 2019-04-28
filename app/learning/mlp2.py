# coding=utf-8
import os
import sys
import tensorflow as tf
import numpy as np
import time
from app.learning.ground_truths import beta_gt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def addLayer(inputs, inSize, outSize, activ_func = None):#insize outsize表示输如输出层的大小，inputs是输入。activ_func是激活函数，输出层没有激活函数。默认激活函数为空
    with tf.name_scope(name = "layer"):
        with tf.name_scope("weigths"):
            weights = tf.Variable(tf.random_normal([inSize,outSize]),name = "W")
        bias = tf.Variable(tf.zeros([1,outSize]),name = "bias")
        w_plus_b = tf.matmul(inputs,weights)+bias
        w_plus_b = tf.nn.dropout(w_plus_b, keep_prob)
        if activ_func == None:
            return w_plus_b
        else:
            return activ_func(w_plus_b)



# mlp全部全连接

vertex_num = 7366 #17436
n_input = 10
learning_rate = 1e-4
keep_probability = 0.9 # 0.2 is good
batch_size = 10
n_hidden = 20

model_path = '../data/model/shape2/test.ckpt'


def shape_graph():
    global keep_prob, x_shape_input, y_shape_true, shape_final_full, shape_cross_entropy, shape_train_step, shape_accuracy

    # 数据输入占位符
    keep_prob = tf.placeholder(tf.float32)
    x_shape_input = tf.placeholder(tf.float32, shape=[None, 10], name="betas")
    y_shape_true = tf.placeholder(tf.float32, shape=[None, vertex_num * 3], name="displacements")

    # 第1、2、3层: 全连接
    full1 = addLayer(x_shape_input, 10, n_hidden, tf.nn.relu)
    shape_final_full = addLayer(full1, n_hidden, vertex_num * 3)

    # 损失优化、评估准确率
    shape_cross_entropy = tf.reduce_mean(tf.square(y_shape_true - shape_final_full))
    shape_train_step = tf.train.AdamOptimizer(learning_rate).minimize(shape_cross_entropy)
    shape_accuracy = tf.reduce_mean(tf.cast(shape_cross_entropy, 'float'))


def shape_train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100000):
            batch = [beta_gt.betas[0:17], np.reshape(beta_gt.displacement[0:17], (-1, vertex_num * 3))]
            if i % 10 == 0:
                train_accuracy = shape_accuracy.eval(session=sess, feed_dict={x_shape_input: batch[0], y_shape_true: batch[1], keep_prob: 1})
                print('step {}, training accuracy: {}'.format(i, train_accuracy))

            shape_train_step.run(session=sess, feed_dict={x_shape_input: batch[0], y_shape_true: batch[1], keep_prob: keep_probability})
        saver = tf.train.Saver()
        saver.save(sess, model_path)

    writer = tf.summary.FileWriter('logfile', tf.get_default_graph())
    writer.close()


def predict(sess, betas):
    return sess.run(shape_final_full, feed_dict={x_shape_input: [betas], keep_prob: 1})


def dupl(a, times):
    b = []
    for i in range(times):
        b.append(a)
    return np.concatenate(b, 0)


def tst():
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_path)
    for i in range(17):
        res = sess.run(shape_final_full, feed_dict={ x_shape_input: [beta_gt.betas[i]], keep_prob: 1 })
        cost = np.mean(np.square(res.reshape(-1, vertex_num, 3) - beta_gt.displacement[i]))
        print(i, cost)


if __name__ == '__main__':
    shape_graph()
    shape_train()
    tst()