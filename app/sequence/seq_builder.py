from app.geometry.mesh import *
from app.smpl.smpl_np import *
import os
import json


def build_sequence(seq_dir, betas, poses=None, transforms=None):
    if not os.path.exists(seq_dir):
        os.mkdir(seq_dir)
    if poses is None:
        poses = np.zeros((len(betas), 24, 3))
    for i in range(len(betas)):
        if transforms is not None:
            smpl.set_params(beta=betas[i], pose=poses[i], trans=transforms[i])
        else:
            smpl.set_params(beta=betas[i], pose=poses[i])
        mesh = smpl.get_mesh()
        mesh.save(os.path.join(seq_dir, str(i) + '.obj'))


def interpolate_param(param1, param2, frame_num):
    raw = []
    shape = np.shape(param1)
    vec1 = np.array(param1).flatten()
    vec2 = np.array(param2).flatten()
    for i in range(len(vec1)):
        raw.append(np.linspace(vec1[i], vec2[i], frame_num))
    raw = np.transpose(raw)
    res = []
    for r in raw:
        res.append(np.reshape(r, shape))
    return np.array(res)


# if __name__ == '__main__':
#     a = [[1, 2], [1, 2], [1, 2]]
#     b = [[1, 2], [4, 5], [7, 8]]
#     print(interpolate_param(a, b, 10))


def shape_sequences(shapes, frame=5):
    base_dir = './shape'
    if os.path.exists(base_dir):
        if not os.remove(base_dir):
            print('dir exists!')
            return
    os.mkdir(base_dir)
    index = []
    for i in range(len(shapes)):
        betas = interpolate_param(np.zeros((10)), shapes[i], frame)
        seq_dir = os.path.join(base_dir, 'seq_' + str(i))
        build_sequence(seq_dir, betas)
        index.append(seq_dir)
    with open(os.path.join(base_dir + 'index.json'), 'w') as fp:
        json.dump(index, fp)


def build_17_betas_sequence():
    betas = [np.zeros(10)]
    param = [-2, -1, 1, 2]
    for i in range(4):
        for j in range(4):
            vec = np.zeros(10)
            vec[i] = param[j]
            betas.append(vec)
    shape_sequences(betas)


def build_56_poses_sequence():
    pass


if __name__ == '__main__':
    build_17_betas_sequence()