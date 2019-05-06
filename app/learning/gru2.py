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
    n_steps = 5  # 序列长度
    n_hidden = 128  # RNN 单元输出节点个数(即隐藏层个数) 1500
    n_output = 7366 * 3  # 输出
    batch_size = 128  # 小批量大小 128
    iter = 1000  # 迭代次数
    show_step = 10  # 显示步数
    learning_rate = 1e-4  # 学习率
    keep_probability = 0.8
    model = 'model'

    def __init__(self, n_input=24 * 3, n_output=7366 * 3):
        self.n_input = n_input
        self.n_output = n_output

    def build_model(self, layers):
        model = keras.Sequential()

        model.add(keras.layers.GRU(input_dim=layers[0], output_dim=layers[1], activation='tanh', return_sequences=True))
        model.add(keras.layers.Dropout(0.15))  # Dropout overfitting

        # model.add(GRU(layers[2],activation='tanh', return_sequences=True))
        # model.add(Dropout(0.2))  # Dropout overfitting

        model.add(keras.layers.GRU(layers[2], activation='tanh', return_sequences=False))
        model.add(keras.layers.Dropout(0.2))  # Dropout overfitting

        model.add(keras.layers.Dense(output_dim=layers[3]))
        model.add(keras.layers.Activation("linear"))

        start = time.time()
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss="mse", optimizer=sgd)
        model.compile(loss="mse", optimizer="rmsprop")  # Nadam rmsprop
        print("Compilation Time : ", time.time() - start)
        return model

    @property
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

            # 可以看做隐藏层
            model = keras.Sequential()
            model.add(keras.layers.Embedding(input_length=seq_lens))
            model.add(keras.layers.GRU())
            # 取 RNN 最后一个时序的输出，然后经过全连接网络得到输出值
            hidden = tf.nn.dropout(hidden, keep_prob)
            print(np.shape(hidden))
            output = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=self.n_output)
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
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

            # 预测结果评估
            pose_accuracy = cross_entropy  # 求损失

        return graph, x_input, y_true, seq_lens, output, train_step, cross_entropy, keep_prob

    def train(self, ground_truth, model_path=None):
        graph, x_input, y_true, seq_lens, output, train_step, cross_entropy, keep_prob = self.gen_graph

        with tf.Session(graph=graph) as sess:
            # 使用会话执行图
            sess.run(tf.global_variables_initializer())  # 初始化变量

            # 开始迭代 使用Adam优化的随机梯度下降法
            for t in range(self.iter):

                batch = ground_truth.get_batch(self.batch_size)
                batch[0] = np.array(batch[0]).reshape((-1, self.n_steps, self.n_input))
                batch[1] = np.array(batch[1]).reshape((-1, self.n_steps, self.n_output))
                batch[2] = np.array(batch[2]).reshape((-1))

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

    def predict(self, inputs, lengths):
        inputs = np.array(inputs).reshape((-1, self.n_steps, self.n_input))
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
