from lbac.sequence.pose_translator import AmcTranslator
import numpy as np
from com.path_helper import *


def gen_poses(amc_json):
    amc = AmcTranslator().load(amc_json)

    poses = amc.poses_list()

    poses = np.array(poses).tolist()

    save_json(poses, conf_path('temp/amc_poses.json'))


if __name__ == '__main__':
    conf_path('temp/amc_poses.json')
    gen_poses('')
