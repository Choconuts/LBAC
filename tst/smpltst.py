from com.path_helper import *
from com.mesh.mesh import *
from com.posture.smpl import *

import numpy as np

smpl = SMPLModel('../db/model/smpl/male.pkl')

def gen_body():
    pose = np.zeros((24, 3))
    pose[1] = np.array([0, 0, 0.3])
    pose[2] = np.array([0, 0, -0.3])
    smpl.set_params(pose=pose)
    m = Mesh()
    m.vertices = smpl.verts
    m.faces = smpl.faces
    m.update()
    m.save('tpl.obj')


if __name__ == '__main__':
    gen_body()