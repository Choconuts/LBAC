import os
import numpy as np
import random


class GroundTruth:

    def get_batch(self, size):
        return [[0], [0]]

    def get_test(self):
        return [[0], [0]]

    def load(self, gt_file):
        return self

    def save(self, gt_file):
        return self



class SampleId:

    def __init__(self, id, data):
        self.id = id
        self.data = data

    def derefer(self):
        return self.data[self.id]

    def to_list(self):
        return np.array(self.id).tolist()


class BatchManager:

    def __init__(self, max_num, cut):
        self.max_num = max_num
        self.pointer = 0
        self.train_cut = cut
        self.element = np.linspace(0, max_num - 1, max_num)
        self.auto_shufle = False

    def cut(self, new_cut):
        if self.train_cut > new_cut:
            print('warning: smaller new cut!')
        self.train_cut = new_cut
        if self.pointer >= new_cut:
            self.pointer = new_cut - 1
        return self

    def shuffle(self):
        random.shuffle(self.element[0:self.train_cut])
        return self

    def shuffle_all(self):
        random.shuffle(self.element)
        return self

    def get_batch(self, size):
        tmp_ptr = self.pointer
        self.pointer += size
        self.pointer %= self.train_cut
        return self.get_range(tmp_ptr, tmp_ptr + size)

    def get_range(self, start, end):
        res = []
        for i in range(start, end):
            ii = i % self.train_cut
            res.append(self.element[ii])
        return res


if __name__ == '__main__':
    a = np.linspace(0, 10, 11).astype('i')
    random.shuffle(a)
    print(a)

