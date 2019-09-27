from com.learning import mlp, dygru, mlp_old, gru, ground_truth, graph_helper, auto_enc as enc
from absl import app
from absl import flags
import os
from com.path_helper import *
from lbac.train.shape_gt import BetaGroundTruth
from lbac.train.pose_gt import PoseGroundTruth
from com.learning.canvas import Canvas
from tst.tst_gt import TestGroundTruth, TestGroundTruth2
from tst.joint20.skeleton_nn import SKLTGroundTruth
from lbac.train.auto_encoder_mesh_gt import AutoEncoderGroundTruth


beta = BetaGroundTruth
pose = PoseGroundTruth
test = TestGroundTruth
gru_test = TestGroundTruth2
sklt = SKLTGroundTruth
auto = AutoEncoderGroundTruth