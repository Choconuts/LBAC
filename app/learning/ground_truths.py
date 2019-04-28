from app.geometry.mesh import Mesh
from app.geometry.smooth import smooth
from app.smpl.smpl_np import smpl
import numpy as np
from app.geometry.closest_vertex import ClosestVertex
import json
import os

ground_truth_dir = '../data/ground_truths/gt_files/'


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

    def load_template(self):
        self.template = Mesh().load(os.path.join(self.avg_dir, 'avg_smooth.obj'))
        return self


if __name__ == '__main__':
    '''
    重新计算ground-truth，可修改保存位置
    '''
    def load_beta():
        bg = BetaGroundTruth(smooth_times=5)
        bg.beta_dir = '../data/pose_simulation/tst/shape_y_final/'
        bg.load_objs().gen_avg2().calc().save(ground_truth_dir + 'beta_gt_4.json')

    # load_beta()


# 自动加载的ground-truth
beta_gt = BetaGroundTruth().load(ground_truth_dir + 'beta_gt_4.json').load_template()


class PoseGroundTruth:
    pose_ground_truth = {
        # 'poses': [n * array[121 * [24 * 3]]],
        # 'displacements': [n * array[121 * [nv * 3]]]
        # 'sequence_length': [n * 1]
    }

    vertx_relation_file = '../data/ground_truths/relation/vertex_relation_1.json'

    vertex_rela = ClosestVertex().load(vertx_relation_file)

    def get_template_mesh(self):
        Mesh().load(beta_gt.avg_dir + 'avg_smooth.obj')

    def apply(self, weights, v_posed):
        T = np.tensordot(weights, smpl.G, axes=[[1], [0]]) # 6000 * 23 x 23 * 4 * 4 = 6000 * 4 * 4
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        # use slice to get original coordinates
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])
        vh1 = np.copy(v)
        v = v[:, :3]

        return v + smpl.trans.reshape([1, 3])

    def re_transform(self, weights, vertices):
        T = np.tensordot(weights, smpl.G, axes=[[1], [0]])
        v = vertices - smpl.trans.reshape([1, 3]) # 7300 * 3
        v_h = np.hstack((v, np.ones([v.shape[0], 1]))).reshape([-1, 4, 1])  # 7300 * 4 * 1
        T_i =np.linalg.inv(T) # 7300 * 4 * 4
        v_posed = np.matmul(T_i, v_h).reshape([-1, 4])
        v_posed = v_posed[:, :3]
        return v_posed

    def tst(self):
        pose = np.random.rand(24, 3)
        vertices = smpl.verts
        smpl.set_params(pose=pose)
        weights = np.random.rand(6890, 24)
        weights = weights / weights.sum(1).reshape((6890, 1))
        v1 = self.apply(weights, vertices)
        v2 = self.re_transform(weights, v1)

    def gen_truth(self, results_dir):

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

        cloth0 = beta_gt.template
        wait = 21
        stride = 5
        n_pose = 20
        keyframes = []
        for i in range(n_pose):
            keyframes.append(wait + stride * i)

        all_out_poses = []
        all_out_disp = []
        all_out_seqlen = []

        for i in range(23, 24):
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

    def get_all_seq_dirs(self):
        return ['../data/pose_simulation/tst/23']

    def get_objs(self):

        def load_seq(seq_dir, max_frame, min_frame=0):
            def obj_name(i):
                return '%04d' % i + '_00.obj'

            files = os.listdir(seq_dir)
            for i in range(min_frame):
                if obj_name(i) not in files:
                    return []
            objs = []
            for i in range(max_frame):
                if obj_name(i) in files:
                    objs.append(Mesh().load(os.path.join(seq_dir, obj_name(i))))
                else:
                    break
            return objs

        seq_dirs = self.get_all_seq_dirs()
        for seq_dir in seq_dirs:
            objs = load_seq(seq_dir, 121, 0)

    def get_mesh_displacement(self, mesh):
        base = beta_gt.template
        return mesh.vertices - base.vertices

    def save(self, path):
        with open(path, 'w') as fp:
            json.dump(self.pose_ground_truth, fp)
        return self

    def load(self, path):
        with open(path, 'r') as fp:
            self.pose_ground_truth =json.load(fp)
        return self

    def pre_process(self, mesh, pose):
        rela = self.vertex_rela.get_rela()
        body_weights = smpl.weights
        cloth_weights = np.zeros((len(mesh.vertices), 24))
        for i in range(len(mesh.vertices)):
            cloth_weights[i] = body_weights[rela[i]]
        mesh.vertices = self.re_transform(cloth_weights, mesh.vertices)
        mesh.update()
        return mesh, pose


if __name__ == '__main__':
    """
    """
    # PoseGroundTruth().gen_truth('../data/pose_simulation/tst/').save(ground_truth_dir + 'pose_gt_4.json')

pose_gt = PoseGroundTruth().load(ground_truth_dir + 'pose_gt_4.json')


