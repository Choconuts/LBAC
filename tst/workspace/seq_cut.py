from com.path_helper import *
import numpy as np
from com.posture.skeleton import *
from com.posture.smpl import *

import transforms3d
from com.path_helper import *
import math
from com.mesh.mesh import *
from com.posture.smpl import *


angle = 10


def rotate(mat, ax, angle):
    rot = transforms3d.axangles.axangle2mat(ax, math.radians(angle))
    return np.matmul(rot, mat)


def rotate_vector(axangle, axis, angle):

    mat = transforms3d.axangles.axangle2mat(axangle, np.linalg.norm(axangle))

    mat = rotate(mat, axis, angle)

    ax, rad = transforms3d.axangles.mat2axangle(mat)
    ax = ax[:3]
    axangle = ax / np.linalg.norm(ax) * rad
    return axangle


def rotate_arm_and_leg(pose):
    pose[16] = rotate_vector(pose[16], (0, 0, 1), angle)
    pose[17] = rotate_vector(pose[17], (0, 0, 1), -angle)
    pose[1] = rotate_vector(pose[1], (0, 0, 1), angle)
    pose[2] = rotate_vector(pose[2], (0, 0, 1), -angle)


def process_seqs(seqs):
    for seq in seqs:
        for pose in seq:
            rotate_arm_and_leg(pose)


def cut():
    in_file = conf_path('temp/128_r.json')
    v_file = 'valid-20.json'

    valids = load_json(v_file)
    raw_data = load_json(in_file)
    print(len(valids))
    data = []
    for v in valids:
        seq = raw_data[v]
        data.append(seq)

    print(np.array(data).shape)

    return data


def add_zero(data):
    if len(data) == 0:
        return
    data.insert(0, np.zeros((len(data[0]), 24, 3)).tolist())


def gen_new_poses():
    # data = load_json('128_r.json')
    data = cut()
    process_seqs(data)
    add_zero(data)

    smpl = SMPLModel(conf_path('smpl'))
    i = 0
    for pose in data[1]:
        smpl.set_params(np.array(pose))
        smpl.save_to_obj('seq/' + str(i) + '.obj')
        i += 1
        if i > 10:
            break

    out_file = 'pose_seq.json'
    save_json(data, out_file)


if __name__ == '__main__':
    angle = 10
    gen_new_poses()
