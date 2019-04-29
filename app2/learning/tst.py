import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from app2.learning.mlp import MLP
from app2.learning.ground_truth import TstGroundTruth

x = [0, 1, 2, 3, 4]
y = [4, 3, 2, 1, 0]

mlp = MLP(1, 1)
gt = TstGroundTruth()
gt.x = x
gt.y = y
mlp.learning_rate = 1e-2

mlp.train(gt, 'tst/model3')

mlp.load('tst/model3')
r = mlp.predict(np.array([[1.2]]))
print(r)
r = mlp.predict(np.array([[0]]))
print(r)
# keep_prob = tf.placeholder(tf.float32, name="keep_prob")
# x_input = tf.placeholder(tf.float32, shape=[None, 1], name="x")
# y_true = tf.placeholder(tf.float32, shape=[None, 2], name="y")

mlp.load('tst/model3')
mlp.load('tst/model3')
mlp.load('tst/model3')
mlp.load('tst/model3')
r = mlp.predict(np.array([[4]]))
print(r)


if __name__ == '__main__':
    pass