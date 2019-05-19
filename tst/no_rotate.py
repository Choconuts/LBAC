import transforms3d
from com.path_helper import *
import numpy as np
import math
from com.mesh.mesh import *
from com.posture.smpl import *

file = join(conf_path('temp'), 'seq_56.json')


def rotate(mat, ax, angle):
    rot = transforms3d.axangles.axangle2mat(ax, math.radians(angle))
    return np.matmul(rot, mat)


def cancel_rotation(pose):
    axangle = pose[0]
    mat = transforms3d.axangles.axangle2mat(axangle, np.linalg.norm(axangle))

    mat = rotate(mat, (1, 0, 0), -90)
    mat = rotate(mat, (0, 1, 0), -90)

    ax, rad = transforms3d.axangles.mat2axangle(mat)
    ax = ax[:3]
    axangle = ax / np.linalg.norm(ax) * rad
    pose[0] = axangle
    return pose


if __name__ == '__main__':
    obj = load_json(file)

    for poses in obj:
        for i in range(len(poses)):
            poses[i] = cancel_rotation(poses[i])

    save_json(obj, 'seq_56_r.json')



