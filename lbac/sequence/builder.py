from com.mesh.mesh import *
from com.posture.smpl import *
from com.path_helper import *
import os
import json
import itertools
from lbac.sequence.pose_translator import JsonTranslator, PoseTranslator
from lbac.sequence.database_128 import *

default_translator = JsonTranslator()
default_mapper = PoseBetaMapper()


def set_default_translator(translator: PoseTranslator):
    global default_translator
    default_translator = translator


smpl = None

no_log = False

continue_flag = False

frame_log_step = 10


def set_mapper(mapper):
    global default_mapper
    default_mapper = mapper


def set_smpl(smpl_model):
    global smpl
    smpl = smpl_model


def skipping(base_dir):
    skip = 0
    if continue_flag:
        print('continue building in %s' % base_dir)
        skip = find_last_in_dir(base_dir, lambda i: 'seq_' + str5(i)) - 1
        print('skip: %d' % skip)
    return skip


def build_sequence(seq_dir, betas, poses=None, transforms=None):
    """

    :param smpl:
    :param seq_dir: 输出目录，输出模式为0~n.obj
    :param betas: 若数量少于poses，用最后一帧扩展
    :param poses: 若为None，自动取0，和betas一样长
    :param transforms: 通常为0
    :return:
    """
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)
    if poses is None:
        poses = np.zeros((len(betas), 24, 3))

    meta_file = os.path.join(seq_dir, 'meta.json')
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r') as fp:
                obj = json.load(fp)
        except Exception as e:
            if not no_log:
                print(e)
            obj = {}
    else:
        obj = {}
    obj['frames'] = len(poses)
    obj['interp'] = len(betas) - 1
    obj['poses'] = poses.tolist()
    obj['betas'] = betas.tolist()
    obj['beta'] = np.array(betas[-1]).tolist()[0:4]

    skip = 0
    if continue_flag:
        skip = find_last_in_dir(seq_dir, lambda i: str(i) + '.obj') - 1
        if 'restore' not in obj:
            obj['restore'] = []
        obj['restore'].append(skip)

    with open(meta_file, 'w') as fp:
        json.dump(obj, fp)

    for i in range(len(poses)):
        if i < skip:
            continue
        if i > len(betas) - 1:
            beta = betas[len(betas) - 1]
        else:
            beta = betas[i]
        if transforms is not None:
            smpl.set_params(beta=beta, pose=poses[i], trans=transforms[i])
        else:
            smpl.set_params(beta=beta, pose=poses[i])
        mesh = Mesh().from_vertices(smpl.verts, smpl.faces, True)
        mesh.save(os.path.join(seq_dir, str(i) + '.obj'))
        if i % frame_log_step == frame_log_step - 1:
            print('*', end='')

    if not no_log:
        print(': sequence built in ' + seq_dir)


def interpolate_param(param1, param2, frame_num):
    raw = []
    shape = np.shape(param1)
    vec1 = np.array(param1).flatten()
    vec2 = np.array(param2).flatten()
    for i in range(len(vec1)):
        raw.append(np.linspace(vec1[i], vec2[i], frame_num))
    raw = np.transpose(raw)
    res = []
    for r in raw:
        res.append(np.reshape(r, shape))
    return np.array(res)


def shape_sequences(base_dir, shapes, frame=5):
    """

    :param base_dir: 输出文件夹
    :param shapes: 16 * 4
    :param frame: 5
    :return:
    """
    skip = skipping(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for i in range(len(shapes)):
        if i < skip:
            continue
        beta4 =  np.array(shapes[i])
        if len(beta4) < 10:
            beta4 = np.hstack((beta4, np.zeros(10 - len(beta4))))
        betas = interpolate_param(np.zeros((10)), beta4, frame)
        seq_dir = os.path.join(base_dir, 'seq_' + str5(i))
        build_sequence(seq_dir, betas)


def build_17_betas_sequence(out_dir, interp=5):
    with open(conf_path("betas"), 'r') as fp:
        obj = json.load(fp)
    betas = []
    for i in range(17):
        betas.append(obj[i])
    shape_sequences(out_dir, betas, interp)


def pose_sequences(base_dir, pose_beta_pairs):

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    skip = skipping(base_dir)

    i = 0
    for pair in pose_beta_pairs:
        betas = pair[1]
        poses = pair[0]
        # warning
        # conflicts avoiding
        # 这里如果没有continue，也是不会去覆盖已经有的seq的， 但是会重新从头开始
        # 如果有continue，会找到最后一个dir，在其中继续模拟；会跳过前面的
        if not continue_flag:
            while exists(join(base_dir, 'seq_' + str5(i))):
                i += 1
        elif i < skip:
            i += 1
            continue
        seq_dir = os.path.join(base_dir, 'seq_' + str5(i))
        print('building %s with %d frames' % (seq_dir, len(poses)), np.shape(poses), 'of shape ', betas)
        build_sequence(seq_dir, betas, poses)
        i += 1


def build_poses_sequence(out_dir, poses_json, shapes_range, interp=20):
    seqs = default_translator.load(poses_json).poses_list()

    poses_list = []
    for seq in seqs:
        interps = []
        interps.append(interpolate_param(np.zeros((24, 3)), seq[0], interp))
        for frame in seq:
            interps.append(np.array([frame]))
        out = np.concatenate(interps, axis=0)
        poses_list.append(out)

    with open(conf_path("betas"), 'r') as fp:
        obj = json.load(fp)
    shapes = []
    for i in shapes_range:
        shapes.append(obj[i])

    # shape params 0 ~ x
    shapes = np.array(shapes) # 17 * 4
    if len(shapes[0]) < 10:
        shapes = np.hstack((shapes, np.zeros((len(shapes), 10 - len(shapes[0])))))    # 17 * 10

    betas_list = []
    for shape in shapes:
        betas = interpolate_param(np.zeros((10)), shape, interp)
        betas_list.append(betas)

    # prod = itertools.product(poses_list, betas_list)
    default_mapper.poses_list = poses_list
    default_mapper.betas_list = betas_list
    prod = default_mapper.get_pairs()

    pose_sequences(out_dir, prod)


def set_continue(status=True):
    global continue_flag
    continue_flag = status


if __name__ == '__main__':
    """
    """
    from com.posture.smpl import SMPLModel
    # set_smpl(SMPLModel(conf_path('smpl')))
    # build_17_betas_sequence('../../db/temp/shape/sequence')
    x = interpolate_param([1, 2, 3], [4, 5, 6], 3)
    print(x)
