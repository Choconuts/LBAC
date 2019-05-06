from app.main.main import shape_model_path, pose_model_path, beta_gt, pose_gt, smpl, beta_ground_truth, pose_ground_truth, pose_sequences_dir
from app.smpl.smpl_np import SMPLModel
from app.learning.ground_truth import BetaGroundTruth, PoseGroundTruth
from app.geometry.closest_vertex import ClosestVertex
from app.learning.mlp import MLP
from app.learning.gru3 import GRU
from app.display.shader.shaders import *
from app.display.utils import *
from app.configure import *
from app.learning.regressor import *
from app.main.virtual_fitting import VirtualFitting
import numpy as np


class Animation:
    def __init__(self, frame_rate=24):
        self.frame_rate = frame_rate
        self.sequence = []
        self.current_frame = -1
        self.total = 0
        self.start_time = 0

    def push_frame(self, meshes: list):
        self.sequence.append(meshes)

    def play(self):
        if len(self.sequence) == 0:
            return

        self.current_frame = 0
        self.total = len(self.sequence)
        self.start_time = time.time()

    def stop(self):
        self.current_frame = -1

    def current(self):
        if self.current_frame < 0:
            return None
        return self.sequence[self.current_frame]

    def advance(self):
        if self.current_frame < 0:
            return
        past_time = time.time() - self.start_time
        self.current_frame = int(past_time * self.frame_rate)
        self.current_frame %= self.total

    def save(self, out_dir):
        for i in range(len(self.sequence)):
            meshes = self.sequence[i]
            for j in range(len(meshes)):
                meshes[j].save(os.path.join(out_dir, '%04d' % i + '_' + '%04d' % j + '.obj'))

    def load(self, in_dir):
        i = 0
        j = 0
        file = '%04d' % i + '_' + '%04d' % j + '.obj'
        files = os.listdir(in_dir)
        if file not in files:
            return
        self.sequence = []
        meshes = []
        while True:
            mesh = Mesh().load(os.path.join(in_dir, file))
            meshes.append(mesh)
            j += 1
            file = '%04d' % i + '_' + '%04d' % j + '.obj'
            if file not in files:
                self.sequence.append(meshes)
                meshes = []
                i += 1
                j = 0
                file = '%04d' % i + '_' + '%04d' % j + '.obj'
                if file not in files:
                    break



if __name__ == '__main__':
    anima = Animation()
    mlp = MLP().load(shape_model_path)
    gru = GRU(24 * 3, 7366 * 3, 5).load(pose_model_path)
    beta_gt.load(beta_ground_truth).load_template()
    pose_gt.load(pose_ground_truth)
    vertex_rela = ClosestVertex().load(vertex_relation_path)
    vf = VirtualFitting(pose_gt, beta_gt, mlp, gru, vertex_rela)
    for i in range(1):
        vf.pose = pose_gt.pose_seqs[0][60 + i]
        vf.update()
    run_glut(CallBacks())
