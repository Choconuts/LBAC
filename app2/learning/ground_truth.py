import numpy as np
from app2.geometry.smooth import smooth
from app2.geometry.mesh import Mesh
from app2.geometry.closest_vertex import ClosestVertex
from app2.smpl.smpl_np import SMPLModel
from app2.configure import *
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

    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]

    def get_batch_0(self, size):
        assert len(self.x) > 0 and len(self.y) > 0
        xs = []
        ys = []
        for i in range(size):
            xs.append([self.x[i % len(self.x)]])
            ys.append([self.y[i % len(self.y)]])

        return [np.array(xs), np.array(ys)]

    def get_batch(self, size):
        x = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [2, 3, 4, 5, 6]]
        y = [[2, 4, 6, 8, 10], [6, 8, 10, 12, 14], [4, 6, 8, 10, 12]]
        xs = []
        ys = []
        ls = []
        for i in range(size):
            xs.append(x[i % len(x)])
            ys.append(y[i % len(y)])
            ls.append([5])

        return [np.array(xs), np.array(ys), np.array(ls)]


class BetaGroundTruth(GroundTruth):
    """
    gen: init->load_objs->gen_avg->calc->save
    load: load->load_template

    betas_dir: 17个人体按顺序排列，可以是空的因为有一个从模拟结果中提取的函数
    avg_dir: 生产的模板存储位置，其实不是平均而是参数取0的mesh，要是一个路径，因为会放很多东西进去其实
    beta_meta_path: json文件，存储了k个身体参数的列表，默认17
    """
    # 17 * Vn * 3
    beta_ground_truth = {
        # 'betas': [17 * 10],
        # 'displacements': [17 * vn * 3],
    }

    beta_dir = '../data/betas/result_1/'
    avg_dir = '../data/betas/'
    beta_meta_path = '../data/betas/17_betas.json'

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

    def gen_avg(self):
        print('Begin gen')
        beta_smooth_avg = self.beta_smooth_vertices[0]
        m = Mesh().load(self.beta_dir + '0.obj')
        m.vertices = beta_smooth_avg
        m.save(self.avg_dir + 'template.obj')
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
        return self

    def load_template(self, path=None):
        if path:
            self.template = Mesh().load(path)
        else:
            self.template = Mesh().load(os.path.join(self.avg_dir, 'template.obj'))
        return self

    def get_batch(self, size):
        batch = duplicate([self.betas, self.displacement], size)
        return batch

    def extract_result_bodies(self, result_dir):
        i = 0
        i_dir = os.path.join(result_dir, str(i))
        while os.path.isdir(i_dir):
            j = 0
            j_obj = os.path.join(i_dir, "%40d" % j + "_00.obj")
            obj_file = ""
            while os.path.isfile(j_obj):
                obj_file = j_obj
                j += 1
                j_obj = os.path.join(i_dir, "%40d" % j + "_00.obj")
            import shutil

            shutil.copy(obj_file, os.path.join(self.beta_dir, str(i)))
            i += 1
            i_dir = os.path.join(result_dir, str(i))
        return self


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


class PoseGroundTruth(GroundTruth):
    pose_ground_truth = {
        # 'poses': [n * array[121 * [24 * 3]]],
        # 'displacements': [n * array[121 * [nv * 3]]]
        # 'sequence_length': [n * 1]
    }

    def __init__(self, template_mesh_file, smpl):
        self.template = Mesh().load(template_mesh_file)
        self.smpl = smpl

    pose_seqs = np.array([])
    pose_disps = np.array([])
    seq_lengths = np.array([])

    vertex_rela = ClosestVertex().load(vertex_relation_path)

    def apply(self, weights, v_posed):
        T = np.tensordot(weights, self.smpl.G, axes=[[1], [0]]) # 6000 * 23 x 23 * 4 * 4 = 6000 * 4 * 4
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        # use slice to get original coordinates
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])
        vh1 = np.copy(v)
        v = v[:, :3]

        return v + self.smpl.trans.reshape([1, 3])

    def re_transform(self, weights, vertices):
        T = np.tensordot(weights, self.smpl.G, axes=[[1], [0]])
        v = vertices - self.smpl.trans.reshape([1, 3]) # 7300 * 3
        v_h = np.hstack((v, np.ones([v.shape[0], 1]))).reshape([-1, 4, 1])  # 7300 * 4 * 1
        T_i =np.linalg.inv(T) # 7300 * 4 * 4
        v_posed = np.matmul(T_i, v_h).reshape([-1, 4])
        v_posed = v_posed[:, :3]
        return v_posed

    def gen_truth(self, results_dir, wait=21, stride=5, n_pose=20, gen_ranges=None):

        def load_seq_obj(seq_dir, frame):
            def obj_name(i):
                return '%04d' % i + '_00.obj'

            files = os.listdir(seq_dir)
            if obj_name(frame) not in files:
                return None
            else:
                return Mesh().load(os.path.join(seq_dir, obj_name(frame)))

        def get_seq_dir(i):
            return os.path.join(results_dir, str(i))

        def get_seq_poses(i):
            sd = os.path.join('../sequence/pose/', 'seq_' + str(i))
            with open(os.path.join(sd, 'meta.json'), 'r') as fp:
                poses = json.load(fp)
            return poses

        keyframes = []
        for i in range(n_pose):
            keyframes.append(wait + stride * i)

        all_out_poses = []
        all_out_disp = []
        all_out_seqlen = []

        gens = []
        if gen_ranges:
            for rg in gen_ranges:
                gens.extend(list(range(rg[0], rg[1])))
        else:
            gens = [23]

        for i in gens:
            poses = get_seq_poses(i)
            out_meshes = []
            out_poses = [] # 10- * 24 * 3
            for k in keyframes:
                mesh = load_seq_obj(get_seq_dir(i), k)
                if mesh == None:
                    break
                out_meshes.append(mesh)
                out_poses.append(poses[k])
            act_len = len(out_meshes)
            all_out_seqlen.append(act_len)
            for i in range(act_len):
                out_meshes[i], out_poses[i] = self.pre_process(out_meshes[i], out_poses[i])
            out_poses = np.array(out_poses)
            out_poses = np.pad(out_poses, ((0, n_pose - act_len), (0, 0), (0, 0)), 'constant')
            out_poses = out_poses.tolist()

            disps = []
            for mesh in out_meshes:
                disps.append(self.get_mesh_displacement(mesh))
            disps = np.array(disps)
            disps = np.pad(disps, ((0, n_pose - act_len), (0, 0), (0, 0)), 'constant')
            disps = disps.tolist()
            all_out_disp.append(disps)
            all_out_poses.append(out_poses)

        self.pose_ground_truth['poses'] = all_out_poses
        self.pose_ground_truth['displacements'] = all_out_disp
        self.pose_ground_truth['sequence_length'] = all_out_seqlen
        return self

    def get_mesh_displacement(self, mesh):
        base = self.template
        return mesh.vertices - base.vertices

    def save(self, path):
        with open(path, 'w') as fp:
            json.dump(self.pose_ground_truth, fp)
        return self

    def load(self, path):
        with open(path, 'r') as fp:
            self.pose_ground_truth =json.load(fp)
            self.pose_seqs = np.array(self.pose_ground_truth['poses'])
            self.pose_disps = np.array(self.pose_ground_truth['displacements'])
            self.seq_lengths = np.array(self.pose_ground_truth['sequence_length'])
        return self

    def pre_process(self, mesh, pose):
        rela = self.vertex_rela.get_rela()
        body_weights = self.smpl.weights
        cloth_weights = np.zeros((len(mesh.vertices), 24))
        mesh.save('../test/pose_gt0.obj')
        for i in range(len(mesh.vertices)):
            cloth_weights[i] = body_weights[rela[i]]
        self.smpl.set_params(pose=np.array(pose))
        mesh.vertices = self.re_transform(cloth_weights, mesh.vertices)
        mesh.update()
        return mesh, pose

    batch_pointer = 0

    def get_batch(self, size):
        n_seq = len(self.pose_seqs)
        if size > n_seq:
            return duplicate([self.pose_seqs, self.pose_disps, self.seq_lengths], size)
        if size <= n_seq - self.batch_pointer:
            batch = [self.pose_seqs[self.batch_pointer:self.batch_pointer + size],
                     self.pose_disps[self.batch_pointer:self.batch_pointer + size],
                     self.seq_lengths[self.batch_pointer:self.batch_pointer + size]]
            self.batch_pointer += size
            if self.batch_pointer == n_seq:
                self.batch_pointer = 0
            return batch
        batch = [self.pose_seqs[self.batch_pointer:n_seq],
                 self.pose_disps[self.batch_pointer:n_seq],
                 self.seq_lengths[self.batch_pointer:n_seq]]
        rest = size - n_seq + self.batch_pointer # res > 0
        batch[0] = np.concatenate((batch[0], self.pose_seqs[0:rest]), 0)
        batch[1] = np.concatenate((batch[1], self.pose_disps[0:rest]), 0)
        batch[2] = np.concatenate((batch[2], self.seq_lengths[0:rest]), 0)
        self.batch_pointer = rest
        return batch


if __name__ == '__main__':
    a = [[1, 1], [2, 2], [3, 3]]
    c = [[-1], [-2], [-3]]
    b = duplicate([a, c], 7)
    print(b)