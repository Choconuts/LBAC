import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from app.learning.mlp import MLP
from app.learning.gru import GRU
from app.smpl.smpl_np import SMPLModel
from app.configure import *
from app.learning.ground_truth import TstGroundTruth, PoseGroundTruth, BetaGroundTruth

smpl = SMPLModel(smpl_model_path)
# pose_gt = PoseGroundTruth('../../app/data/beta_simulation/avg_smooth.obj', smpl).load('../../app/data/ground_truths/gt_files/pose_gt_4.json')
# beta_gt = BetaGroundTruth().load('../../app/data/ground_truths/gt_files/beta_gt_4.json')

def mlp_tst():
    mlp = MLP() # 20 * 24 * 3, 20 * 7366 * 3
    mlp.batch_size = 1
    mlp.train(beta_gt, "tst/beta_model/1")
    print(np.shape(pose_gt.pose_seqs))
    print(np.shape(pose_gt.pose_disps))
    # mlp.train(pose_gt, "tst/pose_model1")

    print(pose_gt.get_batch(2)[2])

gru = GRU(1, 1, 5)
gru.iter = 10000
gru.batch_size = 200
gru.learning_rate = 1e-2
# gru.train(TstGroundTruth(), 'tst/model/1')
gru.load('tst/model/1')
r = gru.predict([[25, 26, 27, 28, 29]], [5])[0][-1][0][0]
print(r)


if __name__ == '__main__':
    print(list(range(1, 10)))