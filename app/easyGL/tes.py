from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import  numpy as np
from app.easyGL.mesh import *
from app.easyGL.shader import *

rot = 0
def tes_draw():
    global rot
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    shader.draw()

    glPushMatrix()
    glScale(1.5, 1.5, 1.5)
    glRotatef(rot, 0, 1, 0)
    rot += 1
    glBindBuffer(GL_ARRAY_BUFFER, vbo2.id)
    glDrawArrays(GL_TRIANGLES, 0, int(vbo2.num / 7))
    # glBindBuffer(GL_ARRAY_BUFFER, vbo3.id)
    # glDrawArrays(GL_TRIANGLES, 0, int(vbo3.num / 6))
    glPopMatrix()
    glDisableVertexAttribArray(0)  # 解析数据 例如一个矩阵里含有 位置 、颜色、多种信息
    glutSwapBuffers()


def main():
    glutInit([])
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)  # 显示模式 双缓存
    glutInitWindowPosition(100, 100)  # 窗口位置
    glutInitWindowSize(500, 500)  # 窗口大小
    glutInitContextVersion(4,3)   #为了兼容
    glutInitContextProfile(GLUT_CORE_PROFILE)   #为了兼容
    global vbo2
    verts = Mesh().load('./../test/anima/seq1/1.obj').to_vertex_buffer() # './../smpl/17-bodies/1.obj'
    verts = VertexArray(verts).add_cols([1]).get()
    verts2 = Mesh().load('./../smpl/17-bodies/1.obj').to_vertex_buffer()
    verts2 = VertexArray(verts2).add_cols([0]).get()

    glutCreateWindow("sanjiao")  # 创建窗口
    glutDisplayFunc(tes_draw)  # 回调函数
    glutIdleFunc(tes_draw)  # 回调函数

    vbo2 = StaticVBO().bind(np.hstack((verts2, verts)))
    global shader
    shader = SimpleShader().color(0, [1, 0, 0])
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glutMainLoop()


if __name__ == '__main__':
     main()
