from com.learning.ground_truth import *
from com.path_helper import *
import numpy as np, json
from com.mesh.smooth import *


smooth_times = 5
template = Mesh()


def parse_sim(seq_dir):
    meta = load_json(join(seq_dir, 'meta.json'))
    frames = meta['frames']
    beta = meta['beta']
    poses = meta['poses']
    return meta, frames, beta, poses


def parse_ext(ext_dir):
    if exists(join(ext_dir, 'meta.json')):
        meta = load_json(join(ext_dir, 'meta.json'))
        valid_dict = meta['valids']
    else:
        meta = dict()
        valid_dict = dict()
        dirs = os.listdir(ext_dir)
        for seq_dir in dirs:
            count = parse_sim(seq_dir)[1]
            valid_dict[int(seq_dir)] = count
        meta['valids'] = valid_dict
        meta['type'] = 'unknown'

    return meta, valid_dict


def gen_beta_gt_data(ext_dir, gt_dir):

    meta, valid_dict = parse_ext(ext_dir)

    sim_type = meta['config']['type']
    if gt_dir is None:
        gt_dir = join(conf_path('temp'), sim_type)

    vertices = dict()

    for seq_idx in valid_dict:
        frames = valid_dict[seq_idx]
        seq_idx = int(seq_idx)
        mesh = Mesh().load(join(ext_dir, str5(seq_idx), str4(frames - 1) + '.obj'))
        mesh.vertices = smooth_mesh(mesh).vertices
        if seq_idx == 0:
            global template
            template = mesh
        vertices[seq_idx] = mesh.vertices

    data = {
        'betas': {},    # 17 * 4
        'disps': {}     # 17 * 7366 * 3
    }

    meta['index'] = []
    for i in vertices:
        diff = vertices[i] - template.vertices
        seq_meta = load_json(join(ext_dir, str5(i), 'meta.json'))
        data['betas'][i] = seq_meta['beta'][0:4]
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