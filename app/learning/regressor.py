from app.learning.gru3 import GRU
from app.learning.mlp import MLP
import numpy as np


class ShapeRegressor:

    def __init__(self, mlp):
        self.mlp = mlp

    def gen(self, beta):
        return self.mlp.predict([beta])[0].reshape((7366, 3))

    def gens(self, betas):
        return self.mlp.predict(betas)[0].reshape((-1, 7366, 3))


class PoseRegressor:

    def __init__(self, gru):
        self.gru = gru
        self.cache = np.zeros((gru.n_steps, gru.n_input))
        self.cache_ptr = 0

    def push(self, pose):
        pose = np.array(pose).reshape(self.gru.n_input)
        self.cache[self.cache_ptr] = pose
        self.cache_ptr += 1
        self.cache_ptr %= len(self.cache)

    def input(self):
        x = []
        for i in range(self.cache_ptr, self.cache_ptr + self.gru.n_steps):
            x.append(self.cache[i % len(self.cache)])
        return x

    def gen(self, beta, pose):
        self.push(pose)
        res = self.gru.predict([self.input()])[0]
        return np.array(res)[-1].reshape((7366, 3))