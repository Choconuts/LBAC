from com.learning.ground_truth import GroundTruth
import numpy as np
import random

class TestGroundTruth(GroundTruth):

    def load(self, gt_file):
        return self

    def get_batch(self, size):
        diff = 100
        batch = [np.linspace(0, size - 1, size).reshape(-1, 1), np.linspace(diff, size + diff - 1, size).reshape(-1, 1)]
        batch = np.array(batch).transpose([1, 0, 2]).tolist()
        random.shuffle(batch)
        # batch = np.array(batch).transpose([1, 0, 2])
        # for i in range(len(batch[0])):
        #     print(batch[0][i], ' ', end='')
        # print('')
        # for i in range(len(batch[0])):
        #     print(batch[1][i], ' ', end='')
        # print('')
        return [[[0], [1], [2], [3], [4]], [[4], [5], [6], [7], [8]]]
        return [batch[0], batch[1]]

    def get_test(self):
        return [[[43]], [[63]]]


class TestGroundTruth2(GroundTruth):

    def load(self, gt_file):
        return self

    def get_batch(self, size):
        diff = 100
        batch = [np.linspace(0, size - 1, size).reshape(-1, 1), np.linspace(diff, size + diff - 1, size).reshape(-1, 1)]
        batch = np.array(batch).transpose([1, 0, 2]).tolist()
        random.shuffle(batch)
        # batch = np.array(batch).transpose([1, 0, 2])
        # for i in range(len(batch[0])):
        #     print(batch[0][i], ' ', end='')
        # print('')
        # for i in range(len(batch[0])):
        #     print(batch[1][i], ' ', end='')
        # print('')
        return [[[[0], [1], [2]], [[3], [4], [5]]], [[[4], [5], [6]], [[7], [8], [9]]], [3, 3]]
        return [batch[0], batch[1]]

    def get_test(self):
        return [[[[5], [6], [7]]], [[[8], [9], [10]]], [3]]


if __name__ == '__main__':
    [np.linspace(0, 12 - 1, 12).reshape(-1, 1), np.linspace(2, 12 + 1, 12).reshape(-1, 1)]
    batch = [np.linspace(0, 12 - 1, 12).reshape(-1, 1), np.linspace(2, 12 + 1, 12).reshape(-1, 1)]
    batch = np.array(batch).transpose([1, 0, 2]).tolist()
    random.shuffle(batch)
    batch = np.array(batch).transpose([1, 0, 2])
    for i in range(len(batch[0])):
        print(batch[0][i], ' ', end='')
    print('')
    for i in range(len(batch[0])):
        print(batch[1][i], ' ', end='')
    print('')