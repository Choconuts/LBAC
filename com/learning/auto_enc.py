import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from com.learning.graph_helper import GraphBase


# Visualize decoder setting
# Parameters
n_input = 784
n_output = 7366 * 3
learning_rate = 1e-2
decay_step = 500
iter = 1000
keep_probability = 0.8
batch_size = 128
show_step = 100
n_hidden = [100, 5]
test_flag = False           # 测试
save_step = 100
graph_id = 'enc'
display_step = 1
examples_to_show = 10


def graph():

    # tf Graph input (only pictures)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    X = tf.placeholder(tf.float32, shape=[None, n_input])
    code = tf.placeholder(tf.float32, shape=[None, n_hidden[-1]])

    drop = tf.keras.layers.Dropout(1 - keep_prob)
    encoder_op = X
    with tf.variable_scope('encoder'):
        for h in n_hidden:
            dense = tf.keras.layers.Dense(
                name='dense',
                units=h,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.orthogonal,
                activation=tf.nn.leaky_relu,
                # kernel_regularizer=tf.keras.regularizers.l1_l2,
            )
            encoder_op = drop(dense(encoder_op))

        rr = n_hidden.copy()
        rr.reverse()
        rr = rr[1:]

    def decode(c):
        decoder_op = c
        i = 0
        for h in rr:
            dense = tf.keras.layers.Dense(
                name='dense_' + str(i),
                units=h,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.orthogonal,
                activation=tf.nn.leaky_relu,
                # kernel_regularizer=tf.keras.regularizers.l1_l2,
            )
            decoder_op = drop(dense(decoder_op))
            i += 1

        dense = tf.keras.layers.Dense(
            name='dense_' + str(i),
            units=n_input,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.orthogonal,
            activation=None,
            # kernel_regularizer=tf.keras.regularizers.l1_l2,
        )
        decoder_op = drop(dense(decoder_op))
        return decoder_op

    with tf.variable_scope('decoder_t'):
        decoder_tmp = decode(encoder_op)

    with tf.variable_scope('decoder'):
        decoder_op = decode(code)

    # Prediction
    y_pred = decoder_tmp
    # Targets (Labels) are the input data.
    y_true = X

    global_step = tf.Variable(0, trainable=False)
    rate = tf.train.exponential_decay(learning_rate,
                                      global_step=global_step,
                                      decay_steps=decay_step, decay_rate=0.8)

    # Define loss and optimizer, minimize the squared error
    # 比较原始数据与还原后的拥有 784 Features 的数据进行 cost 的对比，
    # 根据 cost 来提升我的 Autoencoder 的准确率
    loss = tf.reduce_mean(tf.square(y_true - y_pred))  # 进行最小二乘法的计算(y_true - y_pred)^2
    # loss = tf.reduce_mean(tf.square(y_true - y_pred))
    optimizer = tf.train.AdamOptimizer(rate).minimize(loss,  global_step=global_step)


    dc = []
    dd = []
    for v in tf.trainable_variables():
        print(v.name)
        vs = v.name.split('/')
        if vs[0] == 'decoder':
            dc.append(v)
        elif vs[0] == 'decoder_t':
            dd.append(v)

    assignment = []
    for i in range(len(dc)):
        assignment.append(tf.assign(dc[i], dd[i]))

    return [optimizer], [X, keep_prob], [encoder_op, decoder_op, code] + assignment + [decoder_tmp], loss, y_pred


def batch_process(batch: list, is_train, local_step):
    batch[0] = np.reshape(batch[0], (-1, n_input))
    if is_train:
        batch.append(keep_probability)
    else:
        batch.append(1)
    return batch


def predict_process(batch: list, is_test):
    batch.append(1)
    return batch


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
    tst()