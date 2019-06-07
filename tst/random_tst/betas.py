from com.learning.canvas import *
from com.learning.graph_helper import *
from com.mesh.mesh import *
from com.mesh.smooth import *
from com.posture.smpl import *
import numpy as np


def gen_random_beta(r=2):
    beta = np.random.rand(4) * r * 2 - r
    print(beta)
    return beta


def batch(beta):
    return [[beta], 1, None]


def init():
    smpl = SMPLModel(conf_path('model/smpl/male.pkl'))
    canvas = Canvas()
    sess = canvas.open('../../db/model/ex/beta/2', 3000)
    g = GraphBase('mlp').restore()
    cloth = Mesh().load('../../db/gt/ex/beta/1/template.obj')
    beta = gen_random_beta(0)
    beta += [-2, 0 ,0 ,0]
    diff = g.predict(sess, batch(beta)).reshape(-1, 3)
    cloth.vertices += diff
    cloth.update()
    beta0 = np.zeros(10)
    beta0[0:4] = beta
    pose0 = np.zeros((24, 3))
    pose0[0] = [1.579, 0, 0]
    trans0 = np.zeros(3)
    trans0 += [0, 0.2, -0.357]
    print(beta0)
    smpl.set_params(beta=beta0, pose=pose0, trans=trans0)
    body = Mesh()
    body.vertices = smpl.verts
    body.faces = smpl.faces
    body.save(conf_path('body.obj', 'tst'))
    cloth.save(conf_path('rebuild.obj', 'tst'))
    canvas.close()


if __name__ == '__main__':
    init()
