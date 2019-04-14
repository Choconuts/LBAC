from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from app.display.camera import Camera
from app.display.objects import OBJ
from app.display.shader.shaders import SimpleShader
import numpy as np
from app.display.vbo import VBO



class Display:

    def __init__(self):
        pass

    def init(self, *args, **kwargs):"""
    固定的参数设置
        :return: 
        """

    def draw(self):"""
    渲染回调
        :return: 
        """

    def resize(self):"""
    改变窗口大小回调
        :return: 
        """


    @staticmethod
    def init_gl():
        glutInit([])
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)  # 显示模式 双缓存
        glutInitContextVersion(4, 3)  # 为了兼容
        glutInitContextProfile(GLUT_CORE_PROFILE)  # 为了兼容

    def run(self):
        pass



if __name__ == '__main__':
    pass
