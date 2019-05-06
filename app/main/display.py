from app.display.shader.shaders import *
from app.display.utils import *
from app.configure import *
from app.learning.regressor import *
from app.main.virtual_fitting import VirtualFitting
from app.main.animation import *
import numpy as np


class View:
    def __init__(self):
        self.lat = 90
        self.lon = 0
        self.scale = 0.5
        self.offset = [0, 0, 0]


views = [View()]


def apply_view(view, rotate=True):
    glTranslatef(view.offset[0], view.offset[1], 0)
    glScalef(view.scale, view.scale, view.scale)
    if rotate:
        glRotatef(view.lat - 90, 1, 0, 0)
        glRotatef(view.lon, 0, 1, 0)


class MouseState:
    def __init__(self):
        self.down = False
        self.x = 0
        self.y = 0
        self.func = 'ROTATE'


mouse_state = MouseState()


def zoom(in_flag):
    if in_flag:
        views[0].scale *= 1.2
    else:
        views[0].scale /= 1.2
    glutPostRedisplay()


def mouse(button, state, x, y):
    mouse_state.down = (state == GLUT_DOWN)
    mouse_state.x = x
    mouse_state.y = y
    if button == 3 or button == 4:
        mouse_state.func = 'SCALE'
        if state == GLUT_UP:
            return
        if button == 3:
            views[0].scale *=1.2
        else:
            views[0].scale /= 1.2
        glutPostRedisplay()
    elif button == GLUT_LEFT_BUTTON:
        mouse_state.func = 'ROTATE'
    elif button == GLUT_RIGHT_BUTTON:
        mouse_state.func = 'TRANSLATE'


def aspect_ratio():
    return glutGet(GLUT_WINDOW_WIDTH) / glutGet(GLUT_WINDOW_HEIGHT)


def directional_light(i, dir, dif):
    dif4 = np.array([dif[0], dif[1], dif[2], 1])
    pos4 = np.array([dir[0], dir[1], dir[2], 0])
    glEnable(GL_LIGHT0 + i)
    glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, dif4)
    glLightfv(GL_LIGHT0 + i, GL_POSITION, pos4)


def ambient_light(a):
    a4 = np.array([a[0], a[1], a[2], 1])
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, a4)


anima = Animation(30)


def draw_mesh(mesh):
    def vertex(v, n):
        glNormal3f(n[0], n[1], n[2])
        glVertex3f(v[0], v[1], v[2])

    glBegin(GL_TRIANGLES)
    for f in mesh.faces:
        def v(i):
            return mesh.vertices[f[i]]

        def n(i):
            return mesh.normal[f[i]]

        for i in range(3):
            vertex(v(i), n(i))
    glEnd()


def draw_room():
    glColor3f(0.8, 0.8, 0.8)
    draw_mesh(anima.current()[1])
    glColor3f(0.7, 0.2, 1)
    draw_mesh(anima.current()[0])
    return


def display():
    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1, 1)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, aspect_ratio(), 0.1, 10)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -1)
    glEnable(GL_LIGHTING)
    glEnable(GL_NORMALIZE)
    directional_light(0, [0, 0, 1], [0.6, 0.6, 0.6])
    ambient_light([0.3, 0.3, 0.3])
    apply_view(views[0])

    draw_room()

    glutSwapBuffers()


def reshape(w, h):
    return


def motion(x, y):
    if not mouse_state.down:
        return
    view = views[0]
    if mouse_state.func == 'ROTATE':
        speed = 0.25
        view.lon += (x - mouse_state.x) * speed
        view.lat += (y - mouse_state.y) * speed
        view.lat = max(-90, min(90, view.lat))
    elif mouse_state.func == 'TRANSLATE':
        speed = 1e-3
        view.offset[0] += (x - mouse_state.x) * speed
        view.offset[1] -= (y - mouse_state.y) * speed
    mouse_state.x = x
    mouse_state.y = y
    glutPostRedisplay()


def run_glut(cb):
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(1280, 720)
    window = glutCreateWindow("ARCSim")
    glutReshapeFunc(reshape)
    glutIdleFunc(cb.idle)
    glutDisplayFunc(display)
    # glutKeyboardFunc(cb.keyboard)
    # glutSpecialFunc(cb.special)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutMainLoop()


class CallBacks:
    def __init__(self):
        self.idle = None


if __name__ == '__main__':
    from app.main.main import shape_model_path, pose_model_path, beta_gt, pose_gt, smpl, beta_ground_truth, \
        pose_ground_truth, pose_sequences_dir
    from app.smpl.smpl_np import SMPLModel
    from app.learning.ground_truth import BetaGroundTruth, PoseGroundTruth
    from app.geometry.closest_vertex import ClosestVertex

    mlp = MLP().load(shape_model_path)
    gru = GRU(24 * 3, 7366 * 3, 5).load(pose_model_path)
    beta_gt.load(beta_ground_truth).load_template()
    pose_gt.load(pose_ground_truth)
    vertex_rela = ClosestVertex().load(vertex_relation_path)
    vf = VirtualFitting(pose_gt, beta_gt, mlp, gru, vertex_rela)
    # for i in range(4):
    #     vf.pose = pose_gt.pose_seqs[0][30 + i]
    #     vf.update()
    # for i in range(5, 60):
    #     vf.pose = pose_gt.pose_seqs[0][30 + i]
    #     vf.update()
    #     anima.push_frame([vf.cloth, vf.body])
    # anima.save('tmp/anima2')
    anima.load('tmp/anima2')
    cb = CallBacks()

    def idle():
        anima.advance()
        glutPostRedisplay()

    cb.idle = idle
    anima.play()
    run_glut(cb)
