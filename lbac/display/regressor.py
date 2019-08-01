from com.learning.graph_helper import *
from com.learning.canvas import *
from com.learning import mlp
from com.learning import gru
import numpy as np
from com.timer import Timer


class ShapeRegressor:

    def __init__(self, canvas, mlp_model_path):
        g2 = GraphBase(mlp.graph_id)
        canvas.open()
        canvas.load_graph(mlp_model_path, g2)
        g2.predict_process = mlp.predict_process
        self.canvas = canvas
        self.mlp = g2

    def gen(self, beta):
        return self.mlp.predict(self.canvas.sess, [[beta[0:4]]])[0].reshape((7366, 3))


class PoseRegressor:

    def __init__(self, canvas, gru_model_path, n_step, load_step=0):
        g = GraphBase(gru.graph_id)
        gru.n_steps = n_step
        gru.n_input = 76
        canvas.open()
        if load_step > 0:
            canvas.load_graph_of_step(gru_model_path, g, load_step)
        else:
            canvas.load_graph(gru_model_path, g)

        g.predict_process = gru.predict_process
        self.gru = g
        self.canvas = canvas
        self.step = n_step
        self.cache = np.zeros((self.step, 72))
        self.cache_ptr = 0

    def push(self, pose):
        pose = np.array(pose).reshape(72)
        self.cache[self.cache_ptr] = pose
        self.cache_ptr += 1
        self.cache_ptr %= len(self.cache)

    def input(self):
        x = []
        for i in range(self.cache_ptr, self.cache_ptr + self.step):
            x.append(self.cache[i % len(self.cache)])
        return x

    def gen(self, beta, pose):
        self.push(pose)
        x = np.array(self.input()).reshape(self.step, 72)
        b = np.zeros((self.step, 1)) + np.array(beta[0:4]).reshape((1, 4))

        res = self.gru.predict(self.canvas.sess, [np.hstack((x, b))])[0]
        return np.array(res).reshape((7366, 3))