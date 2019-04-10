from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from app.display.camera import Camera
from app.display.objects import OBJ

obj = OBJ('./test/', 'save_mesh的副本 4.obj')

class Display:

    camera = Camera()



    def __init__(self):
        pass


def drawFunc():
    glClear(GL_COLOR_BUFFER_BIT)
    glRotatef(1, 0, 1, 0)
    glutWireTeapot(1.5)
    glFlush()


# glutInit()
# glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
# glutInitWindowSize(800, 800)
# glutCreateWindow(b"First")
# glutDisplayFunc(drawFunc)
# glutIdleFunc(drawFunc)
# glutMainLoop()

import sys


window = 0
camera = Camera()


def InitGL(width, height):
    glClearColor(0.1, 0.1, 0.5, 0.1)
    glClearDepth(10.0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
    camera.move(0.0, 0.0, -5)
    obj.create_gl_list()

    # shader_init()


def DrawGLScene():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    camera.setLookat()
    # drawFunc()
    glPushMatrix()
    glScale(5, 5, 5)
    glColor(0.5, 0.5, 0.5)
    glCallList(obj.gl_list)
    glPopMatrix()

    # shader_test()
    glutSwapBuffers()



def ReSizeGLScene(Width, Height):
    glViewport(0, 0, Width, Height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width) / float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def main():
    global window
    # glut init
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    glutInitWindowPosition(36, 2)
    window = glutCreateWindow("display")

    # callbacks
    glutDisplayFunc(DrawGLScene)
    glutIdleFunc(DrawGLScene)
    glutReshapeFunc(ReSizeGLScene)
    glutMouseFunc(camera.mouse_button)
    glutMotionFunc(camera.mouse_move)
    glutKeyboardFunc(camera.keypress)
    glutSpecialFunc(camera.keypress)
    InitGL(640, 480)
    glutMainLoop()


main()


if __name__ == '__main__':
    pass
