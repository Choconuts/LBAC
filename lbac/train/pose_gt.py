from com.learning.ground_truth import *
from com.path_helper import *
import numpy as np, json
from com.mesh.smooth import *


def gen_pose_gt_data(ext_dir, gt_dir, gen_range=None):
    meta = load_json(join(ext_dir, 'meta.json'))
    sim_type = meta['config']['type']
    if gt_dir is None:
        gt_dir = join(conf_path('temp'), sim_type)

    if not exists(gt_dir):
        os.makedirs(gt_dir)


class PoseGroundTruth(GroundTruth):

    def get_batch(self, size):
        return None
