# coding=utf-8
import os
import sys
import tensorflow as tf
import keras
import numpy as np
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def single_layer_dynamic_gru(input_x, n_steps, n_hidden, seq_len):
    '''
    返回动态单层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    input_x = tf.transpose(input_x, [1, 0, 2])
    # 可以看做隐藏层
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    # gru_cell = keras.layers.GRUCell(n_hidden)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    # hiddens, states = keras.layers.RNN(cell=gru_cell, inputs=input_x, dtype=tf.float32, sequence_length=seq_len, time_major=True)
    # hiddens, states = keras.layers.RNN(cell=gru_cell, return_sequences=True, return_state=True, inputs=input_x, dtype=tf.float32, sequence_length=seq_len,
    #                                    time_major=True)

    hiddens, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input_x,
                                       dtype=tf.float32, sequence_length=seq_len, time_major=True)

    # 注意这里输出需要转置  转换为时序优先的
    # hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


class GRU:

    n_input = 24 * 3  # RNN 单元输入节点的个数
    n_steps = 20  # 序列长度
    n_hidden = 128  # RNN 单元输出节点个数(即隐藏层个数) 1500
    n_output = 7366 * 3  # 输出
    batch_size = 128  # 小批量大小 128
    iter = 1000  # 迭代次数
    show_step = 10  # 显示步数
    learning_rate = 1e-4  # 学习率
    decay_step = 200
    keep_probability = 0.8
    model = 'model'

    def __init__(self, n_input=24 * 3, n_output=7366 * 3, n_steps=20):
        self.n_input = n_input
        self.n_output = n_output
        self.n_steps = n_steps

    def gen_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # 定义占位符
            # batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  n_input：表示一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入 RNN 网络
            x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_steps, self.n_input], name="x")
            y_true = tf.placeholder(dtype=tf.float32, shape=[None, self.n_steps, self.n_output], name="y")
            seq_lens = tf.placeholder(tf.int32, shape=[None], name="length")
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # 可以看做隐藏层
            hidden, states = single_layer_dynamic_gru(x_input, self.n_steps, self.n_hidden, seq_lens)

            # 取 RNN 最后一个时序的输出，然后经过全连接网络得到输出值
            hidden = tf.nn.dropout(hidden, keep_prob)
            print(np.shape(hidden))
            output = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=self.n_output, activation_fn=None)
            print(np.shape(output))

            '''
            设置对数似然损失函数
            '''
            # 代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
            y_true_t = tf.transpose(y_true, [1, 0, 2])
            cross_entropy = tf.reduce_mean(tf.square(y_true_t[-1] - output[-1]))

            '''
            求解
            '''
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=self.decay_step, decay_rate=0.8)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

            # 预测结果评估
            pose_accuracy = cross_entropy  # 求损失

        return graph, x_input, y_true, seq_lens, output, train_step, cross_entropy, keep_prob

    def train(self, ground_truth, model_path=None):
        graph, x_input, y_true, seq_lens, output, train_step, cross_entropy, keep_prob = self.gen_graph()

        with tf.Session(graph=graph) as sess:
            # 使用会话执行图
            sess.run(tf.global_variables_initializer())  # 初始化变量

            # 开始迭代 使用Adam优化的随机梯度下降法
            for t in range(self.iter):

                batch = ground_truth.get_batch(self.batch_size)
                batch[0] = np.array(batch[0]).reshape((self.batch_size, -1, self.n_input))
                batch[1] = np.array(batch[1]).reshape((self.batch_size, -1, self.n_output))
                batch[2] = np.array(batch[2]).reshape((self.batch_size))
                batch = self.batch_slice(batch)

                # 开始训练
                train_step.run(feed_dict={x_input: batch[0], y_true: batch[1],
                                          seq_lens: batch[2], keep_prob: self.keep_probability})
                if (t + 1) % self.show_step == 0:
                    # 输出训练集准确率
                    training_cost = sess.run(cross_entropy, feed_dict={x_input: batch[0], y_true: batch[1],
                                                                       seq_lens: batch[2], keep_prob: 1})
                    print(
                        'Step {0}:Training set cost {1}.'.format(t + 1, training_cost))

            # 全部训练完成做测试  分成200次，一次测试50个样本
            # 输出测试机准确率   如果一次性全部做测试，内容不够用会出现OOM错误。所以测试时选取比较小的mini_batch来测试

            #     test_accuracy, test_cost = sess.run([pose_accuracy, pose_cost], feed_dict={pose_input_x: x_batch, pose_input_y: y_batch, pose_seq_len: lenths})
            #     test_accuracy_list.append(test_accuracy)
            #     test_cost_list.append(test_cost)
            #     if (t + 1) % 20 == 0:
            #         print('Step {0}:Test set accuracy {1},cost {2}.'.format(t + 1, test_accuracy, test_cost))
            # print('Test accuracy:', np.mean(test_accuracy_list))
            if model_path:
                saver = tf.train.Saver()
                tf.add_to_collection('output', output)
                saver.save(sess, model_path)

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

    def sequence_slice(self, seq, length):
        res = []
        for i in range(length - self.n_steps + 1):
            res.append(np.array([seq[i:i + self.n_steps]]))

        return np.concatenate(res, 0)

    def batch_slice(self, batch):
        res = [[], [], []]
        for i in range(len(batch[0])):
            res[0].append(self.sequence_slice(batch[0][i], batch[2][i]))
            res[1].append(self.sequence_slice(batch[1][i], batch[2][i]))
        res[0] = np.concatenate(res[0])
        res[1] = np.concatenate(res[1])
        for i in range(len(res[0])):
            res[2].append(self.n_steps)
        return res

    def predict(self, inputs):
        inputs = np.array(inputs)[:, -self.n_steps:].reshape((-1, self.n_steps, self.n_input))
        tf.reset_default_graph()
        with tf.Session() as restore_sess:
            restore_saver = tf.train.import_meta_graph(self.model + '.meta')
            restore_saver.restore(restore_sess, tf.train.latest_checkpoint(os.path.dirname(self.model)))
            g = tf.get_default_graph()
            x_input = g.get_operation_by_name("x").outputs[0]
            seq_lens = g.get_operation_by_name("length").outputs[0]
            keep_prob = g.get_operation_by_name("keep_prob").outputs[0]
            output = g.get_collection("output")
            return restore_sess.run(output, feed_dict={x_input: inputs, seq_lens: [self.n_steps], keep_prob: 1})

    def predict_seq(self, sequence):
        sequence = sequence.reshape((-1, self.n_input))
        inputs = []
        lengths = []
        if len(sequence) < self.n_steps:
            seq = np.pad(sequence, ((self.n_steps - len(sequence), 0), (0, 0)), 'constant')
            inputs.append(seq)
            lengths.append(self.n_steps)
        else:
            for i in range(self.n_steps, len(sequence) + 1):
                inputs.append(sequence[i - 5:i])
                lengths.append(self.n_steps)

        tf.reset_default_graph()
        with tf.Session() as restore_sess:
            restore_saver = tf.train.import_meta_graph(self.model + '.meta')
            restore_saver.restore(restore_sess, tf.train.latest_checkpoint(os.path.dirname(self.model)))
            g = tf.get_default_graph()
            x_input = g.get_operation_by_name("x").outputs[0]
            seq_lens = g.get_operation_by_name("length").outputs[0]
            keep_prob = g.get_operation_by_name("keep_prob").outputs[0]
            output = g.get_collection("output")
            return restore_sess.run(output, feed_dict={x_input: inputs, seq_lens: lengths, keep_prob: 1})
