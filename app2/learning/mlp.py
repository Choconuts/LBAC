# coding=utf-8
import os
import sys
import tensorflow as tf
import numpy as np
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def full_conn_layer(input_layer, output_size, keep_prob=None, activate=None, name=None):
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
    if keep_prob is not None:
        d_fc = tf.nn.dropout(d_fc, keep_prob)
    if activate is not None:
        h_fc = activate(d_fc)        # 激活函数
    else:
        h_fc = d_fc
    h_fc = tf.nn.dropout(h_fc, keep_prob, name=name)
    return h_fc


class MLP:

    n_input = 10
    n_output = 7366 * 3
    learning_rate = 1e-4
    iter = 5000
    show_step = 10
    keep_probability = 0.99
    batch_size = 10
    hidden = [20]
    model = 'model'

    def __init__(self, n_input=10, n_output = 7366 * 3):
        self.n_input = n_input
        self.n_output = n_output

    def gen_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # 数据输入占位符
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            x_input = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x")
            y_true = tf.placeholder(tf.float32, shape=[None, self.n_output], name="y")

            # 第1、2、3层: 全连接
            full = None
            for h in self.hidden:
                full = full_conn_layer(x_input, h, keep_prob, tf.nn.relu)
            output = full_conn_layer(full, self.n_output, keep_prob, name="output")

            # 损失优化、评估准确率
            cross_entropy = tf.reduce_mean(tf.square(y_true - output))
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        return graph, x_input, y_true, output, train_step, cross_entropy, keep_prob

    def train(self, ground_truth, model_path=None):
        graph, x_input, y_true, output, train_step, cross_entropy, keep_prob = self.gen_graph()
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iter):
                batch = ground_truth.get_batch(self.batch_size)
                if i % self.show_step == 0:
                    train_accuracy = cross_entropy.eval(session=sess, feed_dict={x_input: batch[0], y_true: batch[1], keep_prob: 1})
                    print('step {}, training accuracy: {}'.format(i, train_accuracy))
                train_step.run(session=sess, feed_dict={x_input: batch[0], y_true: batch[1], keep_prob: self.keep_probability})

            print(sess.run(output, feed_dict={x_input: np.array([[-2.2]]), keep_prob: 1}))
            if model_path:
                saver = tf.train.Saver()
                tf.add_to_collection('output', output)
                saver.save(sess, model_path)

        return self

        # writer = tf.summary.FileWriter('logfile', tf.get_default_graph())
        # writer.close()

    def load(self, model_path):
        # x_input, y_true, output, train_step, cross_entropy, keep_prob = self.graph()
        # tf.reset_default_graph()
        # restore_graph = tf.Graph()
        # with tf.Session(graph=restore_graph) as restore_sess:
        #     restore_saver = tf.train.import_meta_graph(model_path + '.meta')
        #     restore_saver.restore(restore_sess, tf.train.latest_checkpoint(os.path.dirname(model_path)))
        # self.output = output
        self.model = model_path
        return self

    def predict(self, inputs):
        tf.reset_default_graph()
        with tf.Session() as restore_sess:
            restore_saver = tf.train.import_meta_graph(self.model + '.meta')
            restore_saver.restore(restore_sess, tf.train.latest_checkpoint(os.path.dirname(self.model)))
            g = tf.get_default_graph()
            x_input = g.get_operation_by_name("x").outputs[0]
            keep_prob = g.get_operation_by_name("keep_prob").outputs[0]
            output = g.get_collection("output")
            return restore_sess.run(output, feed_dict={x_input: inputs, keep_prob: 1})
