import numpy as np
from app.smpl.smpl_tf import *
from app.smpl.smpl_np import SMPLModel
import os


def build_p0(betas, dir='.'):

    id = ''
    for b in betas:
        id += '_' + str(b)

    pose = []
    for i in range(24):
        pose.append([0, 0, 0])

    pose = np.array(pose)

    for i in range(10 - len(betas)):
        betas.append(0)

    betas = np.array(betas)
    trans = np.zeros(3)

    pose = tf.constant(pose, dtype=tf.float64)
    betas = tf.constant(betas, dtype=tf.float64)
    trans = tf.constant(trans, dtype=tf.float64)

    output, faces = smpl_model('./model.pkl', betas, pose, trans, True)
    sess = tf.Session()
    result = sess.run(output)

    if not os.path.exists(dir):
        os.mkdir(dir)

    outmesh_path = dir + '/smpl'+ id + '.obj'
    with open(outmesh_path, 'w') as fp:
        for v in result:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))



if __name__ == '__main__':
    for i in [-2, -1, 0, 1, 2]:
        for j in [-2, -1, 0, 1, 2]:
            build_p0([i, j], 'out1')
