from lbac.train.pose_gt import PoseGroundTruth
from lbac.display.regressor import PoseRegressor
from com.learning.canvas import Canvas
from com.sequence.sequence import *
from lbac.display.seq_show import *

pose_gt_dir = ''

test_model_dir = ''

gru_step = 80

load_step = -1

pose_groundtruth = None

index_to_test = 0

pose_regressor = None

canvas = None

model = None


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
    pos_seq.data[:, 17, 2] -= 0.1
    pos_seq.data[:, 17, 1] += 0.1


def predict_pose(pose_seq: Sequence, model_dir=None):
    """
    用model_dir的模型来预测pose的disp，模型会缓存下来，因此下次调用可以不传入model_dir
    :param pose_seq: pose序列
    :param model_dir:
    :return: pose_disp序列
    """
    assert pose_seq.type == 'pose'
    global test_model_dir, pose_regressor
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
    path = '../../tst/tst_seqs/test_pose_18.json'
    model = conf_path('../../LBAC-EXDB/adj-3')
    model = r'D:\Educate\CAD-CG\GitProjects\adj-sm'
    model = conf_path('model/sgru/3')
    model = r'D:\Educate\CAD-CG\GitProjects\s80-2'
    pose_seq = Sequence().load(path)
    pose_seq0 = pose_seq.copy()

    # 显示帧差距
    # pose_seq0.time_step = 2
    # pose_seq.time_step = 2
    # pose_seq.data = pose_seq.data[1:]

    # print(len(pose_seq.data))

    # 序列的测试处理
    turbulent_pos_sequence(pose_seq)
    seq_middle_lerp(pose_seq)
    pose_seq.re_sampling(0.044)
    pose_seq.slice(0.1, 1000)

    rdmpose = np.random.random((24, 3)) * 0.4 - 0.2
    for i in range(pose_seq.get_frame_num()):
        pose_seq.data[i] += np.copy(rdmpose)
    pose_seq.time_step = 0.033

    # 检查关节运动
    # show_pose_seq_joints(pose_seq0)
    # show_multi_pose_seq_joints([pose_seq0, pose_seq])

    disp = predict_pose(pose_seq, model)
    # show_disps(disp)
    print(disp.data)
    show_seqs(pose_disp_seq=disp,
              pose_seq=pose_seq,
              show_body=False)

