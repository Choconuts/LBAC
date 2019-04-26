from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))

def kdtree(point_list, depth=0):
    try:
        k = len(point_list[0]) # assumes all points have the same dimension
    except IndexError as e: # if not point_list:
        return None
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # Sort point list and choose median as pivot element
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2 # choose median

    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )

def find_closest(kdt, point):
    k = len(kdt.location)
    path = []

    def walk(node, depth=0):
        path.append(node.location)
        axis = depth % k

import tensorflow as tf
batch_size = 4
input = tf.random_normal(shape=[3, batch_size, 6], dtype=tf.float32)
cell = tf.nn.rnn_cell.LSTMCell(10, forget_bias=1.0, state_is_tuple=True)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state=init_state, time_major=True)
#time_major如果是True，就表示RNN的steps用第一个维度表示，建议用这个，运行速度快一点。
#如果是False，那么输入的第二个维度就是steps。
#如果是True，output的维度是[steps, batch_size, depth]，反之就是[batch_size, max_time, depth]。就是和输入是一样的
#final_state就是整个LSTM输出的最终的状态，包含c和h。c和h的维度都是[batch_size， n_hidden]



def main():
    """Example usage"""
    point_list = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]
    tree = kdtree(point_list)
    print(tree)
    find_closest(tree, (4, 4))

def tst():
    import numpy as np
    a = [[1, 1, 1], [1, 1, 1]]
    a = np.array(a)
    a = np.pad(a, ((0, 2), (0, 0)), 'constant')
    print(a)

if __name__ == '__main__':
   # main()
   # import numpy as np
   # with tf.Session() as sess:
   #     sess.run(tf.global_variables_initializer())
   #     print(np.shape(sess.run(output)))
   #     print(np.shape(sess.run(final_state)))
   #     print(np.shape(sess.run([output, final_state])))
   tst()
