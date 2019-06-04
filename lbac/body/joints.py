import numpy as np
from com.mesh.mesh import *
from com.path_helper import *
from com.posture.smpl import *
from com.posture.imitator import *
from com.mesh.simple_display import *
from com.mesh.array_renderer import *




def draw_joints():
    smpl = SMPLModel(conf_path('model/smpl/male.pkl'))
    j = smpl.J
    k = smpl.kintree_table.astype('i')

    def draw():
        glColor3f(1, 0.5, 0)
        glPointSize(5)
        glBegin(GL_POINTS)
        for p in j:
            glVertex3d(p[0], p[1], p[2])
        # glColor3f(0, 1, 0.5)
        # p = j[23]
        # glVertex3d(p[0], p[1], p[2])
        glEnd()

    set_display(draw)
    # run_glut()
    with open(conf_path('temp/joints.txt'), 'w') as f:
        for i in range(24):
            idx = k[1][i]
            f.write(str(idx) + ' ')
            for r in range(3):
                f.write(str(j[idx][r]) + ' ')
            f.write(str(k[0][idx]))
            f.write('\n')

    print(k)


def compare():
    smpl = SMPLModel(conf_path('model/smpl/male.pkl'))
    my_obj = Mesh().load(conf_path('temp/my_template.obj'))
    global trans_y
    trans_y = 0.18
    smpl_obj = Mesh()
    pose = np.zeros((24, 3))
    pose[1] = np.array([0, 0, 0.3])
    pose[2] = np.array([0, 0, -0.3])

    def change_body():
        smpl.set_params(trans=np.array([0, trans_y, 0]), pose=pose)
        smpl_obj.vertices = smpl.verts
        smpl_obj.faces = smpl.faces
        smpl_obj.update()
        print("y: ", trans_y)
    change_body()
    # smpl_obj.save('../../tst/smpl-1.obj')

    def draw():
        mr1.render([0.5, 0.5, 0.5])
        mr2.render([0.3, 0.7, 0.8])

    def key(k, x, y):
        global trans_y
        if k == GLUT_KEY_UP:
            trans_y += 0.01
        if k == GLUT_KEY_DOWN:
            trans_y -= 0.01
        change_body()
        glutPostRedisplay()

    mr1 = MeshRenderer(my_obj)
    mr2 = MeshRenderer(smpl_obj)

    set_display(draw)
    set_callbacks(spec=key)
    set_init(init_array_renderer)
    run_glut()


def save_vfw():
    smpl = SMPLModel(conf_path('model/smpl/male.pkl'))
    smpl.set_params(trans=np.array([0, 0.1, 0]))
    smpl_obj = Mesh()
    smpl_obj.vertices = smpl.verts
    smpl_obj.faces = smpl.faces
    smpl_obj.update()

    vo = '../../tst/vfw/vertices.txt'
    fo = '../../tst/vfw/faces.txt'
    wo = '../../tst/vfw/weights.txt'
    jo = '../../tst/vfw/joints.txt'

    with open(vo, 'w') as fp:
        for i in range(len(smpl_obj.vertices)):
            v = smpl_obj.vertices[i]
            for j in range(3):
                fp.write(str(v[j]) + ' ')
            fp.write('\n')

    with open(fo, 'w') as fp:
        for i in range(len(smpl_obj.faces)):
            f = smpl_obj.faces[i]
            for j in range(3):
                fp.write(str(f[j]) + ' ')
            fp.write('\n')

    with open(wo, 'w') as fp:
        for i in range(len(smpl.weights)):
            w = smpl.weights[i]
            for j in range(24):
                fp.write(str(w[j]) + ' ')
            fp.write('\n')




if __name__ == '__main__':
    compare()