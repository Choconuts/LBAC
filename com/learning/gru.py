import tensorflow as tf
import numpy as np
from com.learning.graph_helper import Graph


n_input = 10                # RNN 单元输入节点的个数
n_steps = 5                 # 序列长度
n_hidden = 20               # RNN 单元输出节点个数(即隐藏层个数) 1500
n_output = 10               # 输出
batch_size = 128            # 小批量大小 128
iter = 1000                 # 迭代次数
show_step = 10              # 显示步数
learning_rate = 1e-3        # 学习率
decay_step = 200            # 学习率下降
keep_probability = 0.9      # 保持率
graph_id = 'gru'            # id


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


def graph():
    # 定义占位符
    # batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  n_input：表示一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入 RNN 网络
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input], name="x")
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_output], name="y")
    seq_lens = tf.placeholder(tf.int32, shape=[None], name="length")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # 可以看做隐藏层
    hidden, states = single_layer_dynamic_gru(x_input, n_steps, n_hidden, seq_lens)

    # 取 RNN 最后一个时序的输出，然后经过全连接网络得到输出值
    hidden = tf.nn.dropout(hidden, keep_prob)
    print(np.shape(hidden))
    output = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=n_output, activation_fn=None)
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
    rate = tf.train.exponential_decay(learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_step, decay_rate=0.8)
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy, global_step=global_step)

    # 预测结果评估
    pose_accuracy = cross_entropy  # 求损失

    return [train_step], [x_input, keep_prob, y_true, seq_lens], None, cross_entropy, output[-1] # TODO


def sequence_slice(seq, length):
    res = []
    for i in range(length - n_steps + 1):
        res.append(np.array([seq[i:i + n_steps]]))

    return np.concatenate(res, 0)


def batch_slice(batch):
    res = [[], [], []]
    for i in range(len(batch[0])):
        res[0].append(sequence_slice(batch[0][i], batch[2][i]))
        res[1].append(sequence_slice(batch[1][i], batch[2][i]))
    res[0] = np.concatenate(res[0])
    res[1] = np.concatenate(res[1])
    for i in range(len(res[0])):
        res[2].append(n_steps)
    return res


def batch_process(batch: list, is_train, local_step):
    if len(batch[0]) != batch_size or len(batch[0][0]) <= n_steps:
        return None
    batch[0] = np.array(batch[0]).reshape((batch_size, -1, n_input))
    batch[1] = np.array(batch[1]).reshape((batch_size, -1, n_output))
    batch[2] = np.array(batch[2]).reshape((batch_size))
    if batch[2][0] < n_steps:
        return None
    batch = batch_slice(batch)
    if batch is None:
        return None

    if is_train:
        batch.insert(1, keep_probability)
    else:
        batch.insert(1, 1)

    return batch


def predict_process(batch: list, is_test):
    if len(batch[0]) <= n_steps:
        return None
    batch[0] = np.array(batch[0])[:, -n_steps:].reshape((-1, n_steps, n_input))
    batch.insert(1, 1)
    batch.insert(2, None)
    batch.insert(3, [n_steps])
    return batch[0:4]


def bind(g: Graph):
    g.batch_process = batch_process
    g.show_steps = show_step
    g.iter = iter
    g.batch_size = batch_size


class GRUGraph(Graph):
    def __init__(self):
        Graph.__init__(self, graph_id)
        bind(self)
        self.graph_func = graph


if __name__ == '__main__':
    """
    """
