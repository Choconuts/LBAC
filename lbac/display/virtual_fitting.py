from lbac.train.pose_gt import PoseGroundTruth
from lbac.train.shape_gt import BetaGroundTruth
from com.posture.smpl import apply as apply_pose, SMPLModel
from com.timer import Timer
from com.mesh.closest_vertex import ClosestVertex
from com.mesh.mesh import Mesh
import numpy as np
from lbac.display.regressor import *


class VirtualFitting:

    def __init__(self):

        self.beta_gt = BetaGroundTruth().load(conf_path('beta_gt'))
        self.cloth = Mesh(self.beta_gt.template)
        self.beta = np.zeros(10)
        self.pose = np.zeros((24, 3))
        self.trans = np.zeros(3)
        self.last_beta = np.zeros(10)
        self.shaped_cloth = None
        self.smpl = SMPLModel(conf_path('smpl'))
        vs, fs = self.smpl.set_params(self.pose, self.beta, self.trans)
        self.body = Mesh().from_vertices(vs, fs)
        # 暂存0状态normal
        self.cloth_base_normal = self.cloth.normal
        self.body_base_normal = self.body.normal
        self.canvas = [Canvas(), Canvas()]
        self.shape_regressor = ShapeRegressor(self.canvas[0], conf_path('mlp'))
        self.pose_regressor = PoseRegressor(self.canvas[1], conf_path('gru'), conf_value('gru_step'))
        self.reset_cloth()
        self.relation = ClosestVertex().load(conf_path('vert_rela'))
        self.cloth_weights = np.zeros((len(self.cloth.vertices), 24))
        rela = self.relation.get_rela()
        for i in range(len(self.cloth.vertices)):
            self.cloth_weights[i] = self.smpl.weights[rela[i]]

    def reset_cloth(self, body_flag=True):
        if self.shaped_cloth is None or (self.last_beta != self.beta).any():
            self.shaped_cloth = Mesh(self.beta_gt.template)
            self.shaped_cloth.vertices += self.shape_regressor.gen(self.beta)
            self.last_beta = self.beta
        self.cloth.set_vertices(self.shaped_cloth.vertices)
        self.cloth.normal = np.copy(self.cloth_base_normal)
        if body_flag:
            self.body.normal = np.copy(self.body_base_normal)

    def apply_pose(self):
        self.cloth.vertices = apply_pose(self.smpl, self.cloth_weights, self.cloth.vertices)

    def apply_normal(self, body_flag=True):
        self.cloth.normal = apply_pose(self.smpl, self.cloth_weights, self.cloth.normal)
        if body_flag:
            self.body.normal = apply_pose(self.smpl, self.smpl.weights, self.body.normal)

    def post_processing(self):
        cloth = self.cloth
        body = self.body
        rela = self.relation.calc_rela_once(cloth, body)

        def spread(ci, vec, levels=40):
            past = {}
            bias = 0.002

            # def get_vec(i, level):
            #     return vec * bias / (np.linalg.norm(cloth.vertices[i] - cloth.vertices[ci]) + bias)
            #     # return vec * level / tot_level

            def forward(i, level):
                if level == 0 or i in past:
                    return
                past[i] = 1
                cloth.vertices[i] += vec * bias / (np.linalg.norm(cloth.vertices[i] - cloth.vertices[ci]) + bias)
                for ni in cloth.edges[i]:
                    forward(ni, level - 1)

            forward(ci, levels)

        # to the close skin
        for i in range(len(cloth.vertices)):
            vc = cloth.vertices[i]
            vb = body.vertices[rela[i]]
            bn = body.normal[rela[i]]
            if np.dot(vb - vc, bn) > 0:
                spread(i, bn * np.dot(vb - vc, bn) * 0.5, 20)

        for r in range(2000):
            flag = True
            for i in range(len(cloth.vertices)):
                vc = cloth.vertices[i]
                vb = body.vertices[rela[i]]
                bn = body.normal[rela[i]]
                if np.dot(vb - vc, bn) > 0.004:
                    spread(i, bn * np.dot(vb - vc, bn) * 1, 10)
                    flag = False
            if flag:
                break

        cloth.update_normal_only()
        for i in range(len(cloth.vertices)):
            cn = cloth.normal[i]
            bn = body.normal[rela[i]]
            cloth.vertices[i] += cn * 0.004 + bn * 0.004
        self.cloth = cloth

    def post_processing_2(self):
        cloth = self.cloth
        body = self.body
        rela = self.relation.calc_rela_once(cloth, body)

        # to the close skin
        for i in range(len(cloth.vertices)):
            vc = cloth.vertices[i]
            vb = body.vertices[rela[i]]
            bn = body.normal[rela[i]]
            cloth.vertices[i] += bn * (np.dot(vb - vc, bn) + 0.01)

        # cloth.update_normal_only()
        # for i in range(len(cloth.vertices)):
        #     cn = cloth.normal[i]
        #     bn = body.normal[rela[i]]
        #     cloth.vertices[i] += cn * 0.004 + bn * 0.004

        self.cloth = cloth

    def update(self):
        timer = Timer(False)
        self.smpl.set_params(self.pose, self.beta, self.trans, True)
        self.body.set_vertices(self.smpl.verts)
        # self.body.update_normal_only()
        timer.tick('gen smpl body')
        self.reset_cloth()
        timer.tick('copy template cloth and apply beta displacement')
        self.cloth.vertices += self.pose_regressor.gen(self.beta, self.pose)
        timer.tick('apply pose displacement')
        # self.relation.reset()
        self.apply_pose()
        # self.apply_normal()
        timer.tick('apply pose')
        self.relation.update(self.cloth, self.body)
        self.post_processing()
        self.cloth.update_normal_only()
        timer.tick('postprocessing')

    def update_cloth_only(self):
        timer = Timer()
        self.smpl.set_params(self.pose, self.beta, self.trans, True, True)
        timer.tick('smpl')
        self.reset_cloth(False)
        timer.tick('reset')
        # self.cloth.vertices += self.pose_regressor.gen(self.beta, self.pose)
        timer.tick('diff')
        self.apply_pose()
        timer.tick('pose')
        # self.apply_normal(False)
        timer.tick('normal')

    def recalculate_pose_displacement(self):
        self.smpl.set_params(self.pose, self.beta, self.trans, True, True)
        self.reset_cloth(False)
        self.cloth.vertices += self.pose_regressor.gen(self.beta, self.pose)

    def close(self):
        for canvas in self.canvas:
            canvas.close()


