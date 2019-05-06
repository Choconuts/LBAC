from app.geometry.mesh import *
from app.smpl.smpl_np import *
from app.learning.ground_truth import BetaGroundTruth, PoseGroundTruth
from app.geometry.closest_vertex import ClosestVertex
from app.learning.mlp import MLP
from app.learning.gru3 import GRU
from app.display.shader.shaders import *
from app.display.utils import *
from app.sequence.builder import *
from app.configure import *


pose_seq_json_file = "tmp/pose/128_1.json"
betas_sequences_dir = "tmp/shape"
pose_sequences_dir = "tmp/pose"
template_conf_file = "tmp/template.json"
arcsim_exe_path = ""
cloth_mesh_file = "tmp/tshirt.obj"
material_file = "tmp/gray-interlock.json"
simulation_result_dir = "../data/sim"
beta_ground_truth = "../data/betas/gt_1.json"
pose_ground_truth = "../data/poses/gt_5.json"
shape_model_path = "../data/betas/model/2/mlp"
pose_model_path = "../data/poses/model/5/gru"


def pre_generation(pose_build_range=(0, 128)):
    from app.sequence.cmu import build_joints_sequences_json
    # build pose sequences parameters
    build_joints_sequences_json(cmu_data_dir, pose_seq_json_file)

    build_betas_sequences(betas_sequences_dir, 5) # 第12个人体模拟不出来，原因不明，可以用第13个代替（因为接近），实际上只有16个人体
    build_poses_sequences(pose_seq_json_file, pose_build_range, pose_sequences_dir)


def run_simulation(beta_range, pose_range):
    from app.simulation.runner import ARCSimRunner
    runner = ARCSimRunner(template_conf_file, arcsim_exe_path, cloth_mesh_file, material_file,
                          betas_sequences_dir, pose_sequences_dir, simulation_result_dir)
    if beta_range:
        for i in beta_range:
            runner.run_seq(i, {'type': 0, 'end_frame': 30, 'obs_frame': 5})
    if pose_range:
        for i in pose_range:
            runner.run_seq(i, {'type': 1, 'end_frame': 120, 'obs_frame': 120, 'gravity': [0, 0, -9.8]}) # cmu序列的重力方向不一样


beta_gt = BetaGroundTruth()
beta_gt.beta_dir = '../data/betas/result_1/'
beta_gt.avg_dir = '../data/betas/'
beta_gt.beta_meta_path = '../data/betas/17_betas.json'
pose_gt = PoseGroundTruth('../data/betas/template.obj', smpl, pose_sequences_dir)


def data_generation(shape_flag=1, pose_flag=1, rela_flag=0):
    """

    :param option: 0-no, 1-yes
        *vertex relation should change when only shape ground truth changes
    :return:
    """
    if shape_flag:
        beta_gt.extract_result_bodies(os.path.join(simulation_result_dir,
                                                   'shape')).gen_avg().calc().save(beta_ground_truth)
    if shape_flag and rela_flag:
        ClosestVertex().calc(beta_gt.template, smpl.set_params()).save(vertex_relation_path)
    if pose_flag:
        pose_gt.gen_truth(os.path.join(simulation_result_dir, 'pose'),
                          wait=20,
                          stride=1,
                          n_pose=100,
                          gen_ranges=[(23, 24)],
                          no_rotation=True).save(pose_ground_truth)


def train_shape_model():
    beta_gt.load(beta_ground_truth).load_template()

    vertex_num = len(beta_gt.template.vertices)
    mlp = MLP(10, vertex_num * 3)
    mlp.keep_probability = 0.9                      # 0.8 in the paper
    mlp.hidden = [20]                               # [20] in the paper
    mlp.train(beta_gt, shape_model_path)


def train_pose_model():
    pose_gt.load(pose_ground_truth)
    beta_gt.load_template()

    vertex_num = len(beta_gt.template.vertices)
    gru = GRU(24 * 3, vertex_num * 3, 5)            # same as n_pose in ground-truth
    gru.keep_probability = 0.8                      # 0.8 in the paper
    gru.iter = 300
    gru.learning_rate = 1e-2
    gru.decay_step = 100
    gru.batch_size = 16
    gru.n_hidden = 128                              # 1500 in the paper
    gru.train(pose_gt, pose_model_path)


if __name__ == '__main__':
    # pre_generation()
    # run_simulation((0, 17), (0, 128))
    # data_generation(0, 1)
    # train_shape_model()
    train_pose_model()










