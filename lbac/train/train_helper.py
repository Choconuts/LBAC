from com.learning import mlp, gru, ground_truth, graph_helper
from absl import app
from absl import flags
import os
from com.path_helper import *
from lbac.train.shape_gt import BetaGroundTruth
from lbac.train.pose_gt import PoseGroundTruth
from com.learning.canvas import Canvas
from tst.tst_gt import TestGroundTruth


beta = BetaGroundTruth()
pose = PoseGroundTruth()
test = TestGroundTruth()