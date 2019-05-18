﻿from com.mesh.mesh import *
from com.smpl import *
from com.path_helper import *
import os
import json
import itertools

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
        obj['beta'] = np.array(betas[-1]).tolist()[0:4]
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
    """

    :param base_dir: 输出文件夹
    :param shapes: 16 * 4
    :param frame: 5
    :return:
    """
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    for i in range(len(shapes)):
        beta4 =  np.array(shapes[i])
        if len(beta4) < 10:
            beta4 = np.hstack((beta4, np.zeros(10 - len(beta4))))
        betas = interpolate_param(np.zeros((10)), beta4, frame)
        seq_dir = os.path.join(base_dir, 'seq_' + str5(i))
        build_sequence(seq_dir, betas)


def build_17_betas_sequence(out_dir):
    with open(conf_path("betas"), 'r') as fp:
        obj = json.load(fp)
    betas = []
    for i in range(17):
        betas.append(obj[i])
    shape_sequences(out_dir, betas)


def pose_sequences(base_dir, beta_pose_pairs):

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    i = 0
    for pair in beta_pose_pairs:
        betas = pair[i][0]
        poses = pair[i][1]
        seq_dir = os.path.join(base_dir, 'seq_' + str5(i))
        build_sequence(seq_dir, betas, poses)
        i += 1


def build_poses_sequence(out_dir, poses_json, shapes_range, interp=20):
    with open(poses_json, 'r') as fp:
        obj = json.load(fp)
        seqs = np.array(obj)
    poses_list = []
    for seq in seqs:
        interps = []
        interps.append(interpolate_param(np.zeros((24, 3)), seq[0], interp))
        for frame in seq:
            interps.append(frame)
        out = np.concatenate(interps, axis=0)
        poses_list.append(out)
        print(np.shape(out))
    print(np.shape(poses_list))

    with open(conf_path("betas"), 'r') as fp:
        obj = json.load(fp)
    shapes = []
    for i in shapes_range:
        shapes.append(obj[i])

    # shape params 0 ~ x
    shapes = np.array(shapes) # 17 * 4
    if len(shapes[0]) < 10:
        shapes = np.hstack((shapes, np.zeros(len(shapes), 10 - len(shapes[0]))))    # 17 * 10

    betas_list = []
    for shape in shapes:
        betas = interpolate_param(np.zeros((10)), shape, interp)
        betas_list.append(betas)

    prod = itertools.product(betas_list, poses_list)

    pose_sequences(out_dir, prod)


if __name__ == '__main__':
    """
    """
    from com.smpl import SMPLModel
    # set_smpl(SMPLModel(conf_path('smpl')))
    # build_17_betas_sequence('../../db/temp/shape/sequence')
    print(conf_path('betas'))
