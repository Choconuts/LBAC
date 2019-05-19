from com.learning.ground_truth import *
from com.path_helper import *
from com.mesh.smooth import *
from lbac.train.shape_gt import BetaGroundTruth
from com.mesh.closest_vertex import ClosestVertex
from com.posture.smpl import *


beta_gt = BetaGroundTruth()
relation = ClosestVertex()
smpl = SMPLModel(conf_path('smpl'))


def gen_pose_gt_data(ext_dir, beta_dir, gt_dir, gen_range=None):
    meta = load_json(join(ext_dir, 'meta.json'))
    beta_gt.load(beta_dir)
    sim_type = meta['config']['type']
    if gt_dir is None:
        gt_dir = join(conf_path('temp'), sim_type)
    valid_dict = meta['valids']

    if gen_range is None:
        gen_range = []
        for seq_idx in valid_dict:
            gen_range.append(seq_idx)

    valids = valid_dict.keys()

    gen_range = list(set(gen_range).intersection(set(valids)))

    if not exists(gt_dir):
        os.makedirs(gt_dir)

    meta['index'] = dict()
    for seq_idx in gen_range:
        frames = valid_dict[seq_idx]
        seq_meta = load_json(join(ext_dir, str5(seq_idx), 'meta.json'))
        disps = []
        for i in range(frames):
            mesh_i = Mesh().load(join(ext_dir, str5(seq_idx), str4(i)))
            disp = process(mesh_i, seq_meta['beta'], seq_meta['poses'][i])
            disps.append(disp)
        meta['index'][seq_idx] = len(disps)

        data = dict()
        data['disps'] = np.array(disps).tolist()
        data['poses'] = seq_meta['poses']
        data['beta'] = seq_meta['beta']
        save_json(data, join(gt_dir, str5(seq_idx) + '.json'))

    save_json(meta, join(gt_dir, 'meta.json'))


def process(mesh, beta, pose):
    beta_a = np.array(beta)
    relation.load(conf_path('vert_rela'))
    rela = relation.get_rela()
    body_weights = smpl.weights
    cloth_weights = np.zeros((len(mesh.vertices), 24))

    for i in range(len(mesh.vertices)):
        cloth_weights[i] = body_weights[rela[i]]
    smpl.set_params(pose=np.array(pose))
    mesh.vertices = dis_apply(smpl, cloth_weights, mesh.vertices)
    mesh.update()

    disp = mesh.vertices - beta_gt.template.vertices

    beta_id = -1
    for i in beta_gt.data['betas']:
        beta_i = np.array(beta_gt.data['betas'][i])
        if beta_a == beta_i.all():
            beta_id = i
            break

    if beta_id < 0:
        print('warning: missing beta gt')
        return None


    beta_disp = beta_gt.data['disps'][beta_id]

    return disp - beta_disp


class PoseSampleId(SampleId):

    def derefer(self):
        coords = self.data[0][self.id]
        data = self.data[1][coords[0]]
        step = self.data[2]
        p = coords[1]
        poses = np.array(data['poses'][p:p+step]).reshape((step, -1))
        beta = np.array(data['beta'][p:p+step]).reshape((step, -1))
        disps = np.array(data['disps'][p:p+step]).reshape((step, -1))

        x = np.hstack(poses, beta)

        return x, disps


class PoseGroundTruth(GroundTruth):
    def __init__(self, step=5, cut_off=10):
        self.step = step
        self.mapping = []
        self.test_cut = cut_off

    def load(self, gt_dir):
        self.meta = load_json(join(gt_dir, 'meta.json'))
        index = self.meta['index']
        new_index = dict()
        total = 0
        for i in index:
            if index[i] < self.step:
                continue
            new_index[i] = index[i] - self.step + 1
            total += new_index[i]
            for j in range(new_index[i]):
                self.mapping.append((i, j))
        max_num = total
        self.batch_manager = BatchManager(max_num, max_num - self.test_cut)
        self.samples = []
        self.index = new_index

        self.load_data(gt_dir)

        for i in range(len(self.mapping)):
            sample = PoseSampleId(i, (self.mapping, self.data, self.step))
            self.samples.append(sample)

        return self

    def load_data(self, gt_dir):
        self.data = dict()
        for i in self.index:
            self.data[i] = load_json(join(gt_dir, str5(i) + '.json'))

    def get_batch(self, size):
        ids = self.batch_manager.get_batch(size)
        batch = [[], []]
        for id in ids:
            sample = self.samples[id].derefer()
            batch[0].append(sample[0])
            batch[1].append(sample[1])
        return batch


if __name__ == '__main__':
    a = [1, 2, 2]
    b = [2, 3, 5]
    print(list(set(a).intersection(set(b))))