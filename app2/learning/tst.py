import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from app2.learning.mlp import MLP
from app2.learning.gru import GRU
from app2.smpl.smpl_np import SMPLModel
from app2.configure import *
from app2.learning.ground_truth import TstGroundTruth, PoseGroundTruth, BetaGroundTruth

smpl = SMPLModel(smpl_model_path)
pose_gt = PoseGroundTruth('../../app/data/beta_simulation/avg_smooth.obj', smpl).load('../../app/data/ground_truths/gt_files/pose_gt_4.json')
beta_gt = BetaGroundTruth().load('../../app/data/ground_truths/gt_files/beta_gt_4.json')

def mlp_tst():
    mlp = MLP() # 20 * 24 * 3, 20 * 7366 * 3
    mlp.batch_size = 1
    mlp.train(beta_gt, "tst/beta_model/1")
    print(np.shape(pose_gt.pose_seqs))
    print(np.shape(pose_gt.pose_disps))
    # mlp.train(pose_gt, "tst/pose_model1")

    print(pose_gt.get_batch(2)[2])




if __name__ == '__main__':
    print(list(range(1, 10)))