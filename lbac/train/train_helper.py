from com.learning import mlp, gru, ground_truth, graph_helper
from absl import app
from absl import flags
import os
from com.path_helper import *
from lbac.train.shape_gt import BetaGroundTruth
from lbac.train.pose_gt import PoseGroundTruth
from com.learning.canvas import Canvas
from tst.tst_gt import TestGroundTruth, TestGroundTruth2
from tst.joint20.skeleton_nn import SKLTGroundTruth


beta = BetaGroundTruth()
pose = PoseGroundTruth()
test = TestGroundTruth()
gru_test = TestGroundTruth2()
sklt = SKLTGroundTruth()