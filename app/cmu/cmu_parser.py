import numpy as np
import scipy.io as sio
import os
import re

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
    joints = np.transpose(mat.get('joints3D'))
    return joints


if __name__ == '__main__':
    f = find_all_files('..', 'info.mat')
    j = parse_joints(f[0])
    print(j[10])
