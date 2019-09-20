from com.learning.ground_truth import *
from com.path_helper import *
import numpy as np, json
from com.mesh.smooth import *


smooth_times = 5
template = Mesh()
betas = np.zeros(0)
zero_shape_index = -1

ex_dir = r'D:\Educate\CAD-CG\GitProjects\shape'
cloth_dir = 'shapeModel2019-6-19'
topo_file = 'triangles.txt'

faces = np.zeros(0)


def txt_to_array_3(txt_file, dtype):
    with open(txt_file, 'r') as fp:
        s = ' '
        data = []
        while s:
            s = fp.readline()
            if len(s) < 2:
                continue
            values = s.split(' ')
            tri = []
            for i in range(3):
                vi = values[i]
                tri.append(vi)
            data.append(tri)
        data = np.array(data, dtype)
    return data


def txt_to_array(txt_file, dtype):
    with open(txt_file, 'r') as fp:
        s = None
        data = []
        def read_value():
            s = fp.readline()
            if len(s) < 1:
                return None
            values = s.split(' ')
            return values[-1]

        while s is None:
            s = read_value()
        w = int(float(s))
        s = read_value()
        h = int(float(s))

        data = np.zeros((h, w), dtype)
        for i in range(h):
            for j in range(w):
                s = read_value()
                data[i, j] = s
        return data


def load_betas():
    global betas, zero_shape_index
    betas = txt_to_array(join(ex_dir, 'BetaTraining.txt'), 'f')
    for i in range(17):
        if (betas[i] == np.zeros(4)).all():
            zero_shape_index = i


def ex_beta_mesh(i):
    cloth_file = 'cloth_' + str3(i) + '.txt'
    cloth_file = join(ex_dir, cloth_dir, cloth_file)
    vertices = txt_to_array_3(cloth_file, 'f')
    global faces
    if faces is None or len(faces) == 0:
        faces = txt_to_array_3(join(ex_dir, topo_file), 'i')
        faces -= 1
    mesh = Mesh().from_vertices(vertices, faces)
    mesh.update()
    return mesh


def gen_beta_gt_data(gt_dir):
    print(gt_dir)

    vertices = dict()

    for i in range(17):
        mesh = ex_beta_mesh(i)
        mesh.vertices = smooth_mesh(mesh).vertices
        # mesh.save(conf_path('beta_%d.obj' % i, 'tst'))
        if i == zero_shape_index:
            global template
            template = mesh
        vertices[i] = mesh.vertices

    data = {
        'betas': {},    # 17 * 4
        'disps': {}     # 17 * 7366 * 3
    }
    meta = dict()
    meta['index'] = []

    for i in vertices:
        diff = vertices[i] - template.vertices
        data['betas'][i] = betas[i]
        data['disps'][i] = diff.tolist()
        meta['index'].append(i)

    if not exists(gt_dir):
        os.makedirs(gt_dir)

    save_json(meta, join(gt_dir, 'meta.json'))
    save_json(data, join(gt_dir, 'data.json'))
    template.save(join(gt_dir, 'template.obj'))


def smooth_mesh(mesh):
    return smooth(mesh, smooth_times)


class BetaSampleId(SampleId):
    def derefer(self):
        return [self.data['betas'][self.id], self.data['disps'][self.id]]


class BetaGroundTruth(GroundTruth):

    def load(self, gt_dir):
        self.meta = load_json(join(gt_dir, 'meta.json'))
        self.data = load_json(join(gt_dir, 'data.json'))
        self.template = Mesh().load(join(gt_dir, 'template.obj'))
        self.index = self.meta['index']
        max_num = len(self.index)
        self.batch_manager = BatchManager(max_num, max_num)
        self.samples = []

        for i in self.index:
            sample = BetaSampleId(str(i), self.data)
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


def set_smooth_times(i):
    global smooth_times
    smooth_times = i


def gen():
    load_betas()
    set_smooth_times(5)
    gen_beta_gt_data(conf_path('gt/ex/beta/1'))


def tst():
    beta_gt = BetaGroundTruth().load(conf_path('gt/ex/beta/2019-6-19'))

    def pt(i):
        return conf_path('beta_' + str3(i) + '.obj', 'tst')

    print(beta_gt.data['betas'])
    # print(beta_gt.template.vertices.__len__() * 3)

    batch = beta_gt.get_batch(4)
    print(np.shape(batch[0]))
    print(np.shape(batch[1]))

    # for i in range(17):
    #     mesh = Mesh(beta_gt.template)
    #     mesh.vertices += beta_gt.data['disps'][str(i)]
    #     mesh.update_normal_only()
    #     mesh.save(pt(i))


if __name__ == '__main__':
    """
    """
    tst()
    # set_smooth_times(0)
    # mesh = ex_beta_mesh(0)
    # smooth_mesh(mesh)
    # mesh.save(conf_path('smooth.obj', 'tst'))
    # load_betas()
    # gen_beta_gt_data(conf_path('gt/ex/beta/2019-6-19'))
