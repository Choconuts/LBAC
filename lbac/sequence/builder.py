from com.mesh.mesh import *
from com.smpl import *
from com.path_helper import *
import os
import json

smpl = None


def set_smpl(smpl_model):
    smpl = smpl_model


def build_sequence(seq_dir, betas, poses=None, transforms=None):
    """

    :param smpl:
    :param seq_dir: 输出目录，输出模式为0~n.obj
    :param betas: 若数量少于poses，用最后一帧扩展
    :param poses: 若为None，自动取0，和betas一样长
    :param transforms: 通常为0
    :return:
    """
    if os.path.exists(seq_dir):
        if not os.removedirs(seq_dir):
            print('dir exists!')
            return
    os.mkdir(seq_dir)
    if poses is None:
        poses = np.zeros((len(betas), 24, 3))
    for i in range(len(poses)):
        if i > len(betas) - 1:
            beta = betas[len(betas) - 1]
        else:
            beta = betas[i]
        if transforms is not None:
            smpl.set_params(beta=beta, pose=poses[i], trans=transforms[i])
        else:
            smpl.set_params(beta=beta, pose=poses[i])
        mesh = smpl.get_mesh()
        mesh.save(os.path.join(seq_dir, str4(i) + '.obj'))
    with open(os.path.join(seq_dir, 'meta.json'), 'r+') as fp:
        obj = json.load(fp)
        obj['poses'] = poses.tolist()
        obj['betas'] = betas.tolist()
        json.dump(obj, fp)


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


def shape_sequences(base_dir, shapes, frame=5):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    for i in range(len(shapes)):
        betas = interpolate_param(np.zeros((10)), shapes[i], frame)
        seq_dir = os.path.join(base_dir, 'seq_' + str5(i))
        build_sequence(seq_dir, betas)


def build_17_betas_sequence(out_dir):
    with open(conf_path("betas_17"), 'r') as fp:
        obj = json.load(fp)
    betas = []
    for i in range(17):
        betas.append(obj[str(i)])
    shape_sequences(out_dir, betas)


def pose_sequences(betas_list, poses_list):
    base_dir = './pose'
    if os.path.exists(base_dir):
        if not os.remove(base_dir):
            print('dir exists!')
            return
    os.mkdir(base_dir)
    index = []
    for i in range(len(poses_list)):
        if i < len(betas_list):
            betas = betas_list[i]
        else:
            betas = betas_list[len(betas_list) - 1]
        poses = poses_list[i]
        seq_dir = os.path.join(base_dir, 'seq_' + str(i))
        build_sequence(seq_dir, betas, poses)
        index.append(seq_dir)
    with open(os.path.join(base_dir, 'index.json'), 'w') as fp:
        json.dump(index, fp)




def build_56_poses_sequence():
    with open('seqs_56.json', 'r') as fp:
        obj = json.load(fp)
        seqs = np.array(obj)
    poses_list = []
    for seq in seqs:
        interps = []
        last = np.zeros((24, 3))
        interps.append(interpolate_param(last, seq[0], 20))
        last = seq[0]
        j = 0
        for frame in seq:
            interps.append(interpolate_param(last, frame, 2))
            j += 1
            if j % 20 == 0:
                interps.append(interpolate_param(frame, frame, 10))
            if j >= 40:
                break
            last = np.copy(frame)
        out = np.concatenate(interps, axis=0)
        poses_list.append(out)
        print(np.shape(out))
    print(np.shape(poses_list))
    # shape params 0 ~ x
    betas = [np.zeros(10)]
    param = [-2, -1, 1, 2]
    for i in range(4):
        for j in range(4):
            vec = np.zeros(10)
            vec[i] = param[j]
            betas.append(vec)

    # 先生成0 shape的600帧 pose 序列，共56条
    shapes = interpolate_param(np.zeros((10)), betas[0], 4)

    pose_sequences([shapes], poses_list)


def bujiu():
    base = './pose'
    for i in range(56):
        src = os.path.join(os.path.join(base, 'seq_' + str(i)), '21.obj')
        dst = os.path.join(os.path.join(base, 'seq_' + str(i)), '20.obj')
        import shutil
        shutil.copyfile(src, dst)


if __name__ == '__main__':
    """
    """
    from com.smpl import SMPLModel
    set_smpl(SMPLModel(conf_path('smpl')))
    build_17_betas_sequence('../../db/temp/shape/sequence')
