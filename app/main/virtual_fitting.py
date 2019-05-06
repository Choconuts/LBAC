from app.main.main import shape_model_path, pose_model_path, beta_gt, pose_gt, smpl, beta_ground_truth, pose_ground_truth, pose_sequences_dir
from app.smpl.smpl_np import SMPLModel
from app.learning.ground_truth import BetaGroundTruth, PoseGroundTruth
from app.geometry.closest_vertex import ClosestVertex
from app.learning.mlp import MLP
from app.learning.gru3 import GRU
from app.learning.regressor import *
from app.display.shader.shaders import *
from app.display.utils import *
from app.configure import *
import numpy as np


class VirtualFitting:

    def __init__(self, pose_gt: PoseGroundTruth, beta_gt: BetaGroundTruth, mlp: MLP, gru: GRU, vertex_relation: ClosestVertex):

        self.pose_gt = pose_gt.load(pose_ground_truth)
        self.beta_gt = beta_gt.load(beta_ground_truth).load_template()
        self.cloth = Mesh(beta_gt.template)
        self.body = None
        self.beta = np.zeros(10)
        self.pose = np.zeros((24, 3))
        self.trans = np.zeros(3)
        self.last_beta = np.zeros(10)
        self.shaped_cloth = None
        self.smpl = pose_gt.smpl
        self.shape_regressor = ShapeRegressor(mlp)
        self.pose_regressor = PoseRegressor(gru)
        self.relation = vertex_relation

    def reset_cloth(self):
        if self.shaped_cloth is None or (self.last_beta != self.beta).any():
            self.shaped_cloth = Mesh(beta_gt.template)
            self.shaped_cloth.vertices += self.shape_regressor.gen(self.beta)
            self.last_beta = self.beta
        self.cloth = Mesh(self.shaped_cloth)

    def apply_pose(self):
        body_weights = self.smpl.weights
        self.smpl.set_params(self.pose)
        cloth_weights = np.zeros((len(self.cloth.vertices), 24))
        rela = self.relation.get_rela()
        for i in range(len(self.cloth.vertices)):
            cloth_weights[i] = body_weights[rela[i]]
        self.cloth.vertices = self.pose_gt.apply(cloth_weights, self.cloth.vertices) + self.trans
        self.cloth.update()

    def post_processing(self):
        cloth = self.cloth
        body = self.body
        rela = self.relation.get_rela()
        import math

        def get_closest_face(i):
            vc = cloth.vertices[i]
            vb = body.vertices[rela[i]]
            face_map = body.vertex_face_map
            r0 = vc - vb
            min_d = 1000
            min_fn = [0, 0, 0]
            min_f = [0, 0, 0]
            for fi in face_map[rela[i]]:
                f = body.faces[fi]
                fn = cloth.face_norm(f)
                d = np.dot(fn, r0)
                r1 = body.vertices[f[0]] - vb
                r2 = body.vertices[f[1]] - vb
                r3 = body.vertices[f[2]] - vb
                eps = -0.0005
                if np.dot(r1, r0) < eps or np.dot(r2, r0) < eps or np.dot(r3, r0) < eps:
                    continue
                if math.fabs(d) < min_d:
                    min_d = math.fabs(d)
                    min_fn = fn
                    min_f = f
            return [np.dot(min_fn, r0), min_fn, min_f]

        global impact
        impact = []

        def spread(ci, vec, levels=40):
            past = {}
            bias = 0.002

            def get_vec(i, level):
                return vec * bias / (np.linalg.norm(cloth.vertices[i] - cloth.vertices[ci]) + bias)
                # return vec * level / tot_level

            def forward(i, level):
                if level == 0 or i in past:
                    return
                past[i] = 1
                cloth.vertices[i] += get_vec(i, level)
                for ni in cloth.edges[i]:
                    forward(ni, level - 1)

            forward(ci, levels)

        # to the close skin
        for i in range(len(cloth.vertices)):
            # d, n, f = get_closest_face(i)
            vc = cloth.vertices[i]
            vb = body.vertices[rela[i]]
            bn = body.normal[rela[i]]
            cn = cloth.normal[i]
            if np.dot(vb - vc, bn) > 0:
                spread(i, bn * np.dot(vb - vc, bn) * 0.5, 20)

        for r in range(20000):
            flag = True
            for i in range(len(cloth.vertices)):
                # d, n, f = get_closest_face(i)
                vc = cloth.vertices[i]
                vb = body.vertices[rela[i]]
                bn = body.normal[rela[i]]
                cn = cloth.normal[i]
                if np.dot(vb - vc, bn) > 0.001:
                    spread(i, bn * np.dot(vb - vc, bn) * 1, 10)
                    flag = False
            if flag:
                break

        cloth.update()
        for i in range(len(cloth.vertices)):
            # d, n, f = get_closest_face(i)
            vc = cloth.vertices[i]
            vb = body.vertices[rela[i]]
            cn = cloth.normal[i]
            bn = body.normal[rela[i]]
            cloth.vertices[i] += cn * 0.004 + bn * 0.004

        self.cloth = cloth

    def update(self):
        if self.beta is None:
            self.beta = np.zeros(10)
        if self.pose is None:
            self.pose = np.zeros((24, 3))
        if self.trans is None:
            self.trans = np.zeros(3)

        timer = Timer(False)
        self.body = self.smpl.set_params(self.pose, self.beta, self.trans)
        timer.tick('1')
        self.reset_cloth()
        timer.tick('2')
        self.cloth.vertices += self.pose_regressor.gen(self.beta, self.pose)
        timer.tick('3')
        self.relation.reset()
        self.apply_pose()
        timer.tick('4')
        self.relation.update(self.cloth, self.body)
        timer.tick('5')
        self.post_processing()
        timer.tick('6')


mlp = MLP().load(shape_model_path)
gru = GRU(24 * 3, 7366 * 3, 5).load(pose_model_path)
v = VirtualFitting(pose_gt, beta_gt, mlp, gru, smpl)