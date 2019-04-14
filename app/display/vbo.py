from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import  numpy as np
from app.geometry.mesh import Mesh
# from PyOpenGLtoolbox import *
from app.display.shader.shaders import SimpleShader


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
