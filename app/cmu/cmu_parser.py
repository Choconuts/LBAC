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


if __name__ == '__main__':
    find_all_files('..', 'info.mat')