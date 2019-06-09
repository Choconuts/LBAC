from com.learning.mlp import *
from com.learning.ground_truth import *
from com.posture.smpl import *
from com.path_helper import *
import transforms3d as tr

from com.mesh.simple_display import *

def show_joints(joints):
    def d():
        glColor3f(1, 0.5, 0)
        glPointSize(5)
        glBegin(GL_POINTS)
        for j in joints:
            glVertex3d(j[0], j[1], j[2])
        glEnd()

    set_display(d)
    run_glut()


def process_joints(joints, root_rot):
    joints = np.array(joints)

    # 移到原点
    joints -= joints[0]

    # 消除旋转
    g_mat = tr.euler.euler2mat(*root_rot)

    joints = np.matmul(np.linalg.inv(g_mat), joints.transpose()).transpose()

    # 消除缩放
    factor = np.linalg.norm(joints[4] - joints[1]) * 5
    if factor == 0:
        factor = 1e-10
    joints /= factor

    return joints


def apply_random_transform(joints):
    assert joints.shape[-1] == 3
    import random
    mat = tr.euler.euler2mat(random.random() * 6.28, random.random() * 6.28, random.random() * 6.28)
    return np.matmul(mat, joints.transpose()).transpose()


def gen_training_data():
    smpl = SMPLModel(conf_path('smpl'))

    in_file = conf_path('temp/128_r.json')
    seqs = load_json(in_file)
    poses = []
    for seq in seqs:
        for i in range(10):
            if i * 10 > len(seq):
                break
            poses.append(seq[i * 10])

    data = {
        'y': [],
        'x': []
    }

    print(len(poses))
    for pose in poses:
        smpl.set_params(np.array(pose))
        joints = np.copy(smpl.J)
        weights = np.eye(24)
        joints = apply(smpl, weights, np.array(joints))

        joints = process_joints(joints, pose[0])

        for repeat in range(4):
            joints = apply_random_transform(joints)
            x = []
            y = []
            for i in [6, 12, 16, 17]:
                x.append(joints[i].tolist())
            for i in [3, 9, 13, 14]:
                y.append(joints[i].tolist())
            data['x'].append(x)
            data['y'].append(y)

    print(np.array(data['x']).shape)
    print(np.array(data['y']).shape)

    save_json(data, conf_path('gt/sklt/1.json'))


class SKTL_SampleId(SampleId):
    def derefer(self):
        return [self.data['x'][self.id], self.data['y'][self.id]]


class SKLTGroundTruth(GroundTruth):
    def load(self, gt_file):
        self.data = load_json(gt_file)
        self.batch_manager = BatchManager(1250, 1250)
        self.samples = []

        for i in range(1250):
            sample = SKTL_SampleId(i, self.data)
            self.samples.append(sample)

        return self

    def get_batch(self, size):
        ids = self.batch_manager.get_batch(size)
        batch = [[], []]
        for id in ids:
            sample = self.samples[id].derefer()
            batch[0].append(sample[0])
            batch[1].append(sample[1])
        return batch


if __name__ == '__main__':
    """
    """
    gen_training_data()

