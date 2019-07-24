from lbac.train.pose_gt import PoseGroundTruth
from lbac.display.regressor import PoseRegressor
from com.learning.canvas import Canvas
from com.sequence.sequence import *
from lbac.display.seq_show import *

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


def lerp_test():
    data = get_pose_gt_data()
    indices = pose_groundtruth.index


def turbulent_pos_sequence(pos_seq: Sequence):
    """
    扰动序列的右手的旋转一点
    :param pos_seq:
    :return:
    """
    pos_seq.data = pos_seq.data.reshape(-1, 24, 3)
    pos_seq.data[:, 17, 0] += 0.05
    pos_seq.data[:, 17, 2] -= 0.02
    pos_seq.data[:, 17, 1] += 0.01


def predict_pose(pose_seq: Sequence, model_dir=None):
    """
    用model_dir的模型来预测pose的disp，模型会缓存下来，因此下次调用可以不传入model_dir
    :param pose_seq: pose序列
    :param model_dir:
    :return: pose_disp序列
    """
    assert pose_seq.type == 'pose'
    global test_model_dir
    beta = [0, 0, 0, 0]
    if 'beta' in pose_seq.meta:
        beta = np.array(pose_seq.meta['beta'])
    if model_dir and model_dir != test_model_dir:
        test_model_dir = model_dir
        pose_regressor = None

    disp_seq = Sequence(pose_seq.time_step, 'pose_disp')
    disps = []
    for i in range(pose_seq.get_frame_num()):
        disp = predict(beta, pose_seq.data[i])
        disps.append(disp)
    disp_seq.data = disps

    return disp_seq


def seq_middle_lerp(pose_seq: Sequence):
    """
    对pose序列通过插值来生成其他可以操作的序列
    :param pose_seq:
    :return:
    """
    pose_seq.re_sampling(pose_seq.time_step * 0.5)
    pose_seq.data = pose_seq.data[1:]
    pose_seq.re_sampling(pose_seq.time_step * 2)
    pose_seq.data = pose_seq.data[:-1]
    return pose_seq


if __name__ == '__main__':
    path = '../../tst/test_show_pose_seq.json'
    # pose_seq = Sequence().load(path)
    # show_pose_seq_joints(lerp_pose_seq(pose_seq))
    seq = Sequence()
    seq.data = np.linspace([1, 4], [10, 13], 10)
    print(seq_middle_lerp(seq).data)

