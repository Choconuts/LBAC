from app.geometry.mesh import Mesh
from app.geometry.smooth import smooth
import numpy as np
import json

ground_truth_dir = 'gt_files/'


class BetaGroundTruth:
    # 17 * Vn * 3
    beta_ground_truth = {
        # 'betas': None,
        # 'displacements': None,
    }

    beta_dir = '../data/beta_simulation/result/'
    avg_dir = '../data/beta_simulation/'
    with open('../data/beta_simulation/betas.json', 'r') as fp:
        beta_meta = json.load(fp)

    betas = np.array([])
    beta_disp = np.array([])
    beta_vertices = np.array([])
    beta_smooth_vertices = np.array([])
    displacement = np.array([])

    def __init__(self, sample_number=17, smooth_times=5):
        self.beta_k = sample_number
        self.smooth_factor = smooth_times

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
            beta_smooth_vertices.append(smooth(Mesh().load(self.beta_dir + str(i) + '.obj'), self.smooth_factor).save(self.avg_dir + '/smooth/' + str(i) + '.obj').vertices)
        self.beta_smooth_vertices = np.array(beta_smooth_vertices)
        return self

    def gen_avg(self, smooth_flag=True):
        print('Begin gen')
        beta_avg = np.mean(self.beta_vertices, 0)
        beta_smooth_avg = np.mean(self.beta_smooth_vertices, 0)
        m = Mesh().load(self.beta_dir + '0.obj')
        m.vertices = beta_avg
        print(beta_avg)
        m.save(self.avg_dir + 'avg.obj')
        if not smooth_flag:
            return self
        m = Mesh().load(self.beta_dir + '0.obj')
        m.vertices = beta_smooth_avg
        m.save(self.avg_dir + 'avg_smooth.obj')
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
        self.beta_ground_truth['displacement'] = (self.beta_smooth_vertices - Mesh().load(self.avg_dir + file).vertices).tolist()
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


if __name__ == '__main__':
    # 重新计算ground-truth，可修改保存位置
    BetaGroundTruth(smooth_times=50).load_objs().gen_avg2().calc().save(ground_truth_dir + 'beta_gt.json')


# 自动加载的ground-truth
beta_gt = BetaGroundTruth().load(ground_truth_dir + 'beta_gt.json')


# # 17 * Vn * 3
# beta_ground_truth = {
#     'betas': None,
#     'displacements': None,
# }
#
# beta_dir = '../data/beta_simulation/result/'
# with open('../data/beta_simulation/betas.json', 'r') as fp:
#     beta_meta = json.load(fp)
#
# betas = []
# beta_disp = []
# beta_vertices = []
# beta_smooth_vertices = []
# beta_k = 2
# for i in range(beta_k):
#     betas.append(beta_meta[str(i)])
#     beta_vertices.append(Mesh().load(beta_dir + str(i) + '.obj').vertices)
# for i in range(beta_k):
#     beta_smooth_vertices.append(smooth(Mesh().load(beta_dir + str(i) + '.obj'), 5).vertices)
# betas = np.array(betas)
# beta_vertices = np.array(beta_vertices)
# beta_smooth_vertices = np.array(beta_smooth_vertices)
# beta_avg = np.mean(beta_vertices, 0)
# beta_smooth_avg = np.mean(beta_smooth_vertices, 0)
# m = Mesh().load(beta_dir + '12.obj')
# m. vertices = beta_avg
# m.save('../test/avg_mesh.obj')
#
# m = Mesh().load(beta_dir + '12.obj')
# m. vertices = beta_smooth_avg
# m.save('../test/avg_mesh_smooth.obj')
# print(betas)
# print(np.shape(beta_vertices))


