from OpenGL.GL import *
import numpy as np
import time


class VertexArray:

    def __init__(self, array):
        self.va = array
        self.vn = int(len(array) / 6)
        self.va = np.reshape(self.va, (self.vn, 6))

    def add_cols(self, vec):
        vec = np.array(vec, np.float32)
        cols = np.zeros((self.vn, np.shape(vec)[0]), np.float32)
        cols += vec
        self.va = np.hstack((self.va, cols))
        return self

    def get(self):
        return self.va.flatten()


class VBO:
    id = -1

    def __init__(self):
        self.id = glGenBuffers(1)


class StaticVBO(VBO):
    vertices = np.array([])
    num = 0

    def bind(self, vertices):
        glBindBuffer(GL_ARRAY_BUFFER, self.id)  # 绑定
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)  # 输入数据
        self.num = len(vertices)
        return self


class Timer:
    def __init__(self, prt=True):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.print = prt

    def tick(self, msg=''):
        res = time.time() - self.last_time
        if self.print:
            print(msg, res)
        self.last_time = time.time()
        return res

    def tock(self, msg=''):
        if self.print:
            print(msg, time.time() - self.last_time)
        return time.time() - self.last_time
