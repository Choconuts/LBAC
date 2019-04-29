import numpy as np
from app2.geometry.smooth import smooth
from app2.geometry.mesh import Mesh
import json
import os

class GroundTruth:

    def get_batch(self, size):
        return [[0], [0]]

    def load(self, gt_file):
        return self

    def save(self, gt_file):
        return self


class TstGroundTruth(GroundTruth):

    x = []
    y = []

    def get_batch(self, size):
        assert len(self.x) > 0 and len(self.y) > 0
        xs = []
        ys = []
        for i in range(size):
            xs.append([self.x[i % len(self.x)]])
            ys.append([self.y[i % len(self.y)]])

        return [np.array(xs), np.array(ys)]


class BetaGroundTruth(GroundTruth):
    # 17 * Vn * 3
    beta_ground_truth = {
        # 'betas': [17 * 10],
        # 'displacements': [17 * vn * 3],
    }

    beta_dir = '../data/beta_simulation/result/'
    avg_dir = '../data/beta_simulation/'
    beta_meta_path = '../data/beta_simulation/betas.json'

    betas = np.array([])
    beta_disp = np.array([])
    beta_vertices = np.array([])
    beta_smooth_vertices = np.array([])
    displacement = np.array([])

    def __init__(self, sample_number=17, smooth_times=5):
        self.beta_k = sample_number
        self.smooth_factor = smooth_times
        with open(self.beta_meta_path, 'r') as fp:
            self.beta_meta = json.load(fp)

    def load_objs(self, smooth_flag=True):
        print('Begin load')
        betas = []
        beta_vertices = []
        beta_smooth_vertices = []
        for i in range(self.beta_k):
            betas.append(self.beta_meta[str(i)])
            beta_vertices.append(Mesh().load(self.beta_dir + str(i) + '.obj').vertices)
        self.betas = np.array(betas)
        self.beta_vertices = np.array(beta_vertices)
        if not smooth_flag:
            return self
        for i in range(self.beta_k):
            print('body: ', i)
            beta_smooth_vertices.append(smooth(Mesh().load(self.beta_dir + str(i) + '.obj'), self.smooth_factor).save(
                self.avg_dir + '/smooth/' + str(i) + '.obj').vertices)
        self.beta_smooth_vertices = np.array(beta_smooth_vertices)
        return self

    def gen_avg2(self):
        print('Begin gen')
        beta_smooth_avg = self.beta_smooth_vertices[0]
        m = Mesh().load(self.beta_dir + '0.obj')
        m.vertices = beta_smooth_avg
        m.save(self.avg_dir + 'avg_smooth.obj')
        return self

    def calc(self, smooth_flag=True):
        print('Begin calc')
        self.beta_ground_truth['betas'] = self.betas.tolist()
        if smooth_flag:
            file = 'avg_smooth.obj'
        else:
            file = 'avg.obj'
        self.beta_ground_truth['displacement'] = (
                    self.beta_smooth_vertices - Mesh().load(self.avg_dir + file).vertices).tolist()
        return self

    def save(self, path):
        with open(path, 'w') as fp:
            json.dump(self.beta_ground_truth, fp)
        return self

    def load(self, path):
        with open(path, 'r') as fp:
            self.beta_ground_truth = json.load(fp)
        self.betas = np.array(self.beta_ground_truth['betas'])
        self.displacement = np.array(self.beta_ground_truth['displacement'])
        # print(np.shape(self.betas))
        # print(np.shape(self.displacement))
        # print(self.betas)
        # print(self.displacement)
        return self

    def load_template(self):
        self.template = Mesh().load(os.path.join(self.avg_dir, 'avg_smooth.obj'))
        return self

    def get_batch(self, size):
        batch = duplicate([self.betas, self.displacement], size)
        return batch


def duplicate(arrays, n_row):
    b = []
    n = len(arrays[0])
    a = []
    for arr in arrays:
        a.append(np.array(arr))
        b.append([])
    import random, time
    random.seed(time.time())
    for i in range(n_row):
        idx = random.randint(0, n - 1)
        for j in range(len(a)):
            b[j].append([a[j][idx]])

    for j in range(len(b)):
        b[j] = np.concatenate(b[j], 0)

    return b


if __name__ == '__main__':
    a = [[1, 1], [2, 2], [3, 3]]
    c = [[-1], [-2], [-3]]
    b = duplicate([a, c], 7)
    print(b)