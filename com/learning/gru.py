import tensorflow as tf
import numpy as np
from com.learning.graph_helper import GraphBase


n_input = 10                # RNN 单元输入节点的个数
n_steps = 5                 # 序列长度
n_hidden = [128]             # RNN 单元输出节点个数(即隐藏层个数) 1500
n_output = 10               # 输出
batch_size = 128            # 小批量大小 128
iter = 1000                 # 迭代次数
show_step = 10              # 显示步数
learning_rate = 1e-3        # 学习率
decay_step = 200            # 学习率下降
keep_probability = 0.9      # 保持率
test_flag = False           # 测试
save_step = 100             # saving
graph_id = 'gru'            # id
l1_regular_scale = 0          # l1 正则化
l2_regular_scale = 0          # l2 正则化


def single_layer_static_gru(input_x, n_steps, n_hidden):
    '''
    返回静态单层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
    input_x1 = tf.unstack(input_x, num=n_steps, axis=1)

    # 换一种实现
    # gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    gru_cell = tf.keras.layers.GRUCell(
        units=n_hidden,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.orthogonal,
        dropout=1 - keep_probability,
        recurrent_dropout=1 - keep_probability,
        recurrent_initializer=tf.keras.initializers.orthogonal,
        kernel_regularizer=tf.keras.regularizers.l1_l2,
        bias_regularizer=tf.keras.regularizers.l1_l2
    )
    # gru_cell = tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell, output_keep_prob=keep_probability)
    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
    hiddens, states = tf.contrib.rnn.static_rnn(cell=gru_cell, inputs=input_x1, dtype=tf.float32)

    return hiddens, states


def single_layer_dynamic_gru(input_x, n_steps, n_hidden, seq_len):
    '''
    返回静态单层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
    # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度

    # input_x1 = tf.unstack(input_x, num=n_steps, axis=1)
    input_x1 = tf.transpose(input_x, [1, 0, 2])

    # 换一种实现
    # gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    gru_cell = tf.keras.layers.GRUCell(
        units=n_hidden,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.orthogonal,
        dropout=1 - keep_probability,
        recurrent_dropout=1 - keep_probability,
        recurrent_initializer=tf.keras.initializers.orthogonal,
        kernel_regularizer=tf.keras.regularizers.l1_l2,
        bias_regularizer=tf.keras.regularizers.l1_l2
    )
    # gru_cell = tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell, output_keep_prob=keep_probability)
    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
    hiddens, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input_x1, dtype=tf.float32, sequence_length=seq_len, time_major=True)

    return hiddens, states


def graph():
    # 定义占位符
    # batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  n_input：表示一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入 RNN 网络
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input], name="x")
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_output], name="y")
    seq_lens = tf.placeholder(tf.int32, shape=[None], name="length")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # 可以看做隐藏层
    # hidden, states = single_layer_static_gru(x_input, n_steps, n_hidden[0])
    hidden, states = single_layer_dynamic_gru(x_input, n_steps, n_hidden[0], seq_lens)

    # 取 RNN 最后一个时序的输出，然后经过全连接网络得到输出值
    hidden = tf.nn.dropout(hidden, keep_prob)
    print(np.shape(hidden))
    # output = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=n_output, activation_fn=None)
    dense = tf.keras.layers.Dense(
        units=n_output,
        use_bias=True,
        kernel_initializer=tf.keras.initializers.orthogonal,
        kernel_regularizer=tf.keras.regularizers.l1_l2,
    )
    output = dense(hidden)
    print(np.shape(output))
    print(tf.get_collection(tf.GraphKeys.WEIGHTS))
    '''
    设置对数似然损失函数
    '''
    # l1_regularizer = tf.contrib.layers.l1_regularizer(l1_regular_scale, scope=None)
    # l2_regularizer = tf.contrib.layers.l2_regularizer(l2_regular_scale, scope=None)
    # regularizer = tf.contrib.layers.sum_regularizer([l1_regularizer, l2_regularizer], scope=None)
    # regular_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=None)

    # 代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    y_true_t = tf.transpose(y_true, [1, 0, 2])
    # cross_entropy = tf.reduce_mean(tf.square(y_true_t[-1] - output[-1]))    # MSE
    cross_entropy = tf.reduce_mean(tf.abs(y_true_t[-1] - output[-1]))    # MAE

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


def batch_process(batch: list, is_train, local_step):
    if len(batch[0]) != batch_size or len(batch[0][0]) < n_steps:
        return None
    batch[0] = np.array(batch[0]).reshape((batch_size, -1, n_input))
    batch[1] = np.array(batch[1]).reshape((batch_size, -1, n_output))

    # batch = batch_slice(batch)
    if batch is None:
        return None

    if is_train:
        batch.insert(1, keep_probability)
    else:
        batch.insert(1, 1)
    batch.append(np.ones((batch_size)) * n_steps)
    return batch


def predict_process(batch: list, is_test):
    if is_test:
        global batch_size
        tmp = batch_size
        batch_size = len(batch[0])
        batch = batch_process(batch, False, 1)
        batch_size = tmp
        return batch
    if len(batch[0]) < n_steps:
        return None
    batch[0] = np.array(batch[0])[-n_steps:].reshape((-1, n_steps, n_input))
    batch.insert(1, 1)
    batch.insert(2, None)
    return batch[0:4]


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
