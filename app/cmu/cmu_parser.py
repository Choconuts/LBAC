import numpy as np
import scipy.io as sio
import os
import re
import json

info = sio.loadmat('info.mat')
joints = info['joints3D']

# print(joints)
# joints = np.transpose(joints)
# print(np.shape(joints))


def find_all_files(base_dir, pattern):
    """
    找到指定路径下的所有满足正则表达式
    :return:
    """
    targets = []
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for file in files:
            if re.match(pattern, file):
                targets.append(os.path.join(root, file))
    return targets


def parse_joints(info_file):
    """

    :param info_file:
    :return: 100 * 24 * 3
    """
    mat = sio.loadmat(info_file)
    joints = np.transpose(mat.get('pose'))
    return np.reshape(joints, (-1, 24, 3))


def reflect(i):
    return str(int(10001 * i / 56)) + '.mat'


if __name__ == '__main__':
    # f = find_all_files('..', 'info.mat')
    # j = parse_joints(f[0])
    info_dir = r'I:\Choconuts\Download\SURREAL_v1\all_infos'
    obj = []
    for i in range(128):
        joints = parse_joints(os.path.join(info_dir, reflect(i)))
        obj.append(joints.tolist())
    print(np.shape(obj))
    with open('seqs_128_of_joints.json', 'w') as fp:
        json.dump(obj, fp)


