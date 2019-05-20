from com.learning.ground_truth import GroundTruth
import numpy as np


class TestGroundTruth(GroundTruth):

    def load(self, gt_file):
        return self

    def get_batch(self, size):
        return [np.linspace(0, size - 1, size).reshape(-1, 1), np.linspace(2, size + 1, size).reshape(-1, 1)]

    def get_test(self):
        return [[[4.3]], [[6.3]]]