# coding=utf-8
import os
import sys
import tensorflow as tf
import numpy as np
import time
from app.learning.ground_truths import beta_gt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from app.geometry.mesh import *
from app.learning.mlp import vertex_num
from app.learning.ground_truths import pose_gt

# input field

outputs_objs_dir = '../data/pose_simulation/tst/'
seqs_info_dir = '../sequence/pose/'


def seq_length(seq_dir):
    files = os.listdir(seq_dir)
    for i in range(121):
        obj = '%04d' % i + '_00.obj'
        if obj not in files:
            return i
    return 121


def load_poses(i):
    return np.zeros((121, 24, 3))


def load_seq(i):
    seq_path = outputs_objs_dir + str(i)
    files = os.listdir(seq_path)
    objs = []
    for i in range(121):
        obj = '%04d' % i + '_00.obj'
        if obj not in files:
            break
        objs.append(Mesh().load(os.path.join(seq_path, obj)))
    return objs


def tst_get_batch(size):
    y = []
    x = []
    for i in range(size):
        pose = load_poses(23)
        seq = load_seq(23)
        y.append(seq)
        x.append(pose)
    y = np.array(y)
    x = np.array(x)
    print('input: ', np.shape(x))
    print('output: ', np.shape(y))


import ssl

ssl._create_default_https_context = ssl._create_unverified_context



def single_layer_dynamic_gru(input_x, n_steps, n_hidden, seq_len):
    '''
    返回动态单层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 可以看做隐藏层
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input_x, dtype=tf.float32, sequence_length=seq_len)

    # 注意这里输出需要转置  转换为时序优先的
    # hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


n_input = 24 * 3  # RNN 单元输入节点的个数
n_steps = 20  # 序列长度
n_hidden = 128  # RNN 单元输出节点个数(即隐藏层个数) 1500
n_disp = vertex_num * 3  # 输出
batch_size = 4  # 小批量大小 128
training_step = 100  # 迭代次数
display_step = 10  # 显示步数
learning_rate = 1e-4  # 学习率
pose_model_path = '../data/model/pose/test1.ckpt'


def pose_graph():
    global pose_input_x, pose_input_y, pose_seq_len, pose_train_step, pose_accuracy, pose_cost, pose_output
    # 定义占位符
    # batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  n_input：表示一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入 RNN 网络
    pose_input_x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input])
    pose_input_y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_disp])
    pose_seq_len = tf.placeholder(tf.int32, shape=[None], name="length")

    # 可以看做隐藏层
    hiddens, states = single_layer_dynamic_gru(pose_input_x, n_steps, n_hidden, pose_seq_len)

    # 取 RNN 最后一个时序的输出，然后经过全连接网络得到输出值
    pose_output = tf.contrib.layers.fully_connected(inputs=hiddens, num_outputs=n_disp)

    '''
    设置对数似然损失函数
    '''
    # 代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    pose_cost = tf.reduce_mean(tf.square(pose_input_y - pose_output))

    '''
    求解
    '''
    pose_train_step = tf.train.AdamOptimizer(learning_rate).minimize(pose_cost)

    # 预测结果评估
    pose_accuracy = pose_cost  # 求损失


def train_pose():
    print('Begin training...')
    # 创建list 保存每一迭代的结果
    test_accuracy_list = []
    test_cost_list = []
    with tf.Session() as sess:
        # 使用会话执行图
        sess.run(tf.global_variables_initializer())  # 初始化变量

        # 开始迭代 使用Adam优化的随机梯度下降法
        for t in range(training_step):

            x_batch = np.zeros((batch_size, n_steps * n_input))
            for i in range(len(x_batch)):
                x_batch[i] = np.array(pose_gt.pose_ground_truth['poses'][0]).reshape((n_steps * n_input))
            y_batch = np.zeros((batch_size, n_steps, n_disp))
            for i in range(len(x_batch)):
                y_batch[i] = np.array(pose_gt.pose_ground_truth['displacements'][0]).reshape((n_steps, n_disp))

            lenths = np.ones((batch_size))
            for i in range(len(x_batch)):
                lenths[i] = pose_gt.pose_ground_truth['sequence_length'][0]

            # Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1, n_steps, n_input])

            # 开始训练
            pose_train_step.run(feed_dict={pose_input_x: x_batch, pose_input_y: y_batch, pose_seq_len: lenths})
            if (t + 1) % display_step == 0:
                # 输出训练集准确率
                training_accuracy, training_cost = sess.run([pose_accuracy, pose_cost],
                                                            feed_dict={pose_input_x: x_batch, pose_input_y: y_batch, pose_seq_len: lenths})
                print('Step {0}:Training set accuracy {1},cost {2}.'.format(t + 1, training_accuracy, training_cost))

        # 全部训练完成做测试  分成200次，一次测试50个样本
        # 输出测试机准确率   如果一次性全部做测试，内容不够用会出现OOM错误。所以测试时选取比较小的mini_batch来测试
        for t in range(200):

            x_batch = np.zeros((2, n_steps * n_input))
            for i in range(len(x_batch)):
                x_batch[i] = np.array(pose_gt.pose_ground_truth['poses'][0]).reshape((n_steps * n_input))
            y_batch = np.zeros((2, n_steps, n_disp))
            for i in range(len(x_batch)):
                y_batch[i] = np.array(pose_gt.pose_ground_truth['displacements'][0]).reshape((n_steps, n_disp))

            lenths = np.ones((2))
            for i in range(len(x_batch)):
                lenths[i] = pose_gt.pose_ground_truth['sequence_length'][0]

            # Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1, n_steps, n_input])
            test_accuracy, test_cost = sess.run([pose_accuracy, pose_cost], feed_dict={pose_input_x: x_batch, pose_input_y: y_batch, pose_seq_len: lenths})
            test_accuracy_list.append(test_accuracy)
            test_cost_list.append(test_cost)
            if (t + 1) % 20 == 0:
                print('Step {0}:Test set accuracy {1},cost {2}.'.format(t + 1, test_accuracy, test_cost))
        print('Test accuracy:', np.mean(test_accuracy_list))
        saver = tf.train.Saver()
        saver.save(sess, pose_model_path)


def predict_pose(sess, poses):
    l = len(poses)
    poses = np.pad(poses, ((0, n_steps - l), (0, 0), (0, 0)), 'constant')
    x = np.array([np.reshape(poses, (n_steps, n_input))])
    return sess.run(pose_output, feed_dict={pose_input_x: x, pose_seq_len: [l]})


if __name__ == '__main__':
    pose_graph()
    train_pose()

# gru field

# train field