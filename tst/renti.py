import numpy as np
from com.path_helper import *
from com.posture.smpl import SMPLModel
from com.mesh.mesh import Mesh
import time


smpl = SMPLModel('../db/model/smpl.pkl')


def gen_body(beta, pose):
    m = Mesh()
    smpl.set_params(beta=beta, pose=pose)
    m.vertices = smpl.verts
    m.faces = smpl.faces
    m.update()
    m.save("tst.obj")


if __name__ == '__main__':
    import numpy.random as nr

    beta = np.zeros(10)
    # beta[:4] = nr.rand((4)) * 4 - 2
    beta[0] = -2
    pose = np.zeros((24, 3))
    pose[1] = np.array([0, 0, 0.2])
    pose[2] = np.array([0, 0, -0.2])
    print(beta)
    gen_body(beta, pose)