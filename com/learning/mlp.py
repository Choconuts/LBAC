import tensorflow as tf
import numpy as np
from com.learning.graph_helper import GraphBase


n_input = 4
n_output = 7366 * 3
learning_rate = 1e-3
decay_step = 500
iter = 1000
keep_probability = 0.99
batch_size = 10
show_step = 100
n_hidden = [20]
test_flag = False           # 测试
save_step = 100
graph_id = 'mlp'


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
    h_fc = tf.nn.dropout(h_fc, keep_prob=keep_prob, name=name)
    return h_fc


def graph():
    # 数据输入占位符
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    x_input = tf.placeholder(tf.float32, shape=[None, n_input], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, n_output], name="y")

    # 第1、2、3层: 全连接
    full = x_input
    for h in n_hidden:
        if h == 0:
            continue
        full = full_conn_layer(x_input, h, keep_prob, tf.nn.relu)
    output = full_conn_layer(full, n_output, keep_prob, name="output")

    global_step = tf.Variable(0, trainable=False)
    rate = tf.train.exponential_decay(learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_step, decay_rate=0.8)

    # 损失优化、评估准确率
    cross_entropy = tf.reduce_mean(tf.square(y_true - output))
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy, global_step=global_step)

    return [train_step], [x_input, keep_prob, y_true], None, cross_entropy, output


def array(batch):
    for i in range(len(batch)):
        batch[i] = np.array(batch[i])


def batch_process(batch: list, is_train, local_step):
    if is_train:
        batch.insert(1, keep_probability)
    else:
        batch.insert(1, 1)
    array(batch)
    batch[0] = batch[0].reshape((-1, n_input))
    batch[2] = batch[2].reshape((-1, n_output))
    return batch


def predict_process(batch: list, is_test):
    batch.insert(1, 1)
    if len(batch) < 3:
        batch.insert(2, None)
    return batch[0:3]


def bind(g: GraphBase):
    g.set_steps(iter, show_step, save_step)
    g.set_callback(batch_process, predict_process)
    g.batch_size = batch_size
    g.graph_func = graph


class Graph(GraphBase):
    def __init__(self):
        GraphBase.__init__(self, graph_id)
        bind(self)


if __name__ == '__main__':
    """
    """

