from lbac.train.pose_gt import PoseGroundTruth
from lbac.display.regressor import PoseRegressor
from com.learning.canvas import Canvas

pose_gt_dir = ''

test_model_dir = ''

gru_step = 5

load_step = -1

pose_groundtruth = None

index_to_test = 0

pose_regressor = None

canvas = None


def get_canvas():
    global canvas
    if canvas is None:
        if load_step >= 0:
            canvas = Canvas()
        else:
            canvas = Canvas()

    return canvas


def predict(beta, pose):
    global pose_regressor
    if pose_regressor is None:
        pose_regressor = PoseRegressor(get_canvas(), test_model_dir, gru_step)

    return pose_regressor.gen(beta, pose)


def get_pose_gt_data():
    global  pose_groundtruth
    if pose_gt_dir is None:
        pose_groundtruth = PoseGroundTruth().load(pose_gt_dir)

    return pose_groundtruth.data


class Sequence:
    def __init__(self, seq_data):
        self.disps = seq_data['disps']
        self.poses = seq_data['poses']
        self.beta = seq_data['beta']


def lerp_test():
    data = get_pose_gt_data()
    indices = pose_groundtruth.index
    seq = Sequence(data[indices[index_to_test]])



