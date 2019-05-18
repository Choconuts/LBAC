import os
import numpy as np


class GroundTruth:

    def get_batch(self, size):
        return [[0], [0]]

    def get_test(self):
        return [[0], [0]]

    def load(self, gt_file):
        return self

    def save(self, gt_file):
        return self


def duplicate(arrays, n_row):
    b = []
    n = len(arrays[0])
    a = []
    for arr in arrays:
        a.append(np.array(arr))
        b.append([])
    import random, time
    random.seed(time.time())
    for i in range(n_row):
        idx = random.randint(0, n - 1)
        for j in range(len(a)):
            b[j].append([a[j][idx]])

    for j in range(len(b)):
        b[j] = np.concatenate(b[j], 0)

    return b
