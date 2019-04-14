from app.display.display import Display
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from app.display.camera import Camera
from app.display.objects import OBJ
from app.display.shader.shaders import SimpleShader
import numpy as np
from app.display.vbo import VBO


class SimpleDisplay(Display):

    def __init__(self):
        Display.__init__(self)
        self.meshes = dict()

        # if len(window) == 4:
        #     x = window[0]
        #     y = window[1]
        #     w = window[2]
        #     h = window[3]
        # elif len(window) == 2:
        #     x = 0
        #     y = 0
        #     w = window[0]
        #     h = window[1]
        # else:
        #     return
        #
        # glutInit([])
        # glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)  # 显示模式 双缓存
        # glutInitWindowPosition(x, y)  # 窗口位置
        # glutInitWindowSize(w, h)  # 窗口大小

    def init(self, *args, **kwargs):
        glutInitWindowPosition(100, 100)  # 窗口位置
        glutInitWindowSize(800, 600)  # 窗口大小
        glClearColor(0.1, 0.1, 0.5, 0.1)
        glClearDepth(10.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        # gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
        # camera.move(0.0, 0.0, -5)

        SimpleShader().color([1, 0, 0])
        glEnable(GL_DEPTH_TEST)

    def draw(self):
        global rot
        if rot is None:
            rot = 0
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        SimpleShader.draw()
        glBindBuffer(GL_ARRAY_BUFFER, vbo2.id)

        glPushMatrix()
        glScale(1.5, 1.5, 1.5)
        glRotatef(rot, 0, 1, 0)
        rot += 1
        glDrawArrays(GL_TRIANGLES, 0, int(vbo2.num / 6))
        glPopMatrix()
        glDisableVertexAttribArray(0)  # 解析数据 例如一个矩阵里含有 位置 、颜色、多种信息
        glutSwapBuffers()

    def add_mesh(self, name, mesh, color):
        self.meshes[name] = [mesh, color]


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
