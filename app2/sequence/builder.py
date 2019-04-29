from app.geometry.mesh import *
from app.smpl.smpl_np import *
import os
import json


def build_sequence(seq_dir, betas, poses=None, transforms=None):
    """

    :param seq_dir: 构建序列的目录，空不空无所谓，但是最好不要有别的序列在，尤其是比该序列长的
    :param betas: 每帧的beta（会自动将最后一帧扩充到和poses一样的长度）
    :param poses: 每帧的pose（如果是None，就变成和betas一样长）
    :param transforms: 整体位移，无用
    :return: None
    """
    if not os.path.exists(seq_dir):
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
        mesh.save(os.path.join(seq_dir, str(i) + '.obj'))
    with open(os.path.join(seq_dir, 'meta.json'), 'w') as fp:
        json.dump(poses.tolist(), fp)


def interpolate_param(param1, param2, frame_num):
    """

    :param param1: 开始的数值
    :param param2: 结束的数值
    :param frame_num: 插值帧数（包括开头结尾）
    :return: 帧集
    """
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
    with open(os.path.join(base_dir, 'index.json'), 'w') as fp:
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


def pose_sequences(betas_list, poses_list, skip):
    base_dir = './pose'
    #if os.path.exists(base_dir):
    #    if not os.remove(base_dir):
    #        print('dir exists!')
    #        return
    #os.mkdir(base_dir)
    index = []
    for i in range(skip, len(poses_list)):
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
    with open('seqs_128.json', 'r') as fp:
        obj = json.load(fp)
        seqs = np.array(obj)
    poses_list = []
    for i in range(1):
        seq = seqs[i]
        interps = []
        last = np.zeros((24, 3))
        interps.append(interpolate_param(last, seq[0], 20))
        j = 0
        last = seq[0]
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

    # 先生成0 shape的120帧 pose 序列，共56条
    shapes = interpolate_param(np.zeros((10)), betas[0], 4)

    pose_sequences([shapes], poses_list, 0)


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
    # build_17_betas_sequence()
    build_56_poses_sequence()
    # bujiu()
