import numpy as np
import itertools


class PoseBetaMapper:

    def __init__(self):
        self.poses_list = []
        self.betas_list = []

    def get_pairs(self):
        return itertools.product(self.poses_list, self.betas_list)


class SpecialMapper(PoseBetaMapper):

    def __init__(self):
        PoseBetaMapper.__init__(self)
        self.start = 0
        self.end = -1

    def get_pairs(self):
        assert len(self.betas_list) == 17
        assert len(self.poses_list) == 129
        b_f9 = self.betas_list[:9]
        b_b8 = self.betas_list[-8:]
        assert len(b_f9) == 9
        assert len(b_b8) == 8

        p_f57 = self.poses_list[:57]
        p_b72 = self.poses_list[-72:]
        assert len(p_f57) == 57
        assert len(p_b72) == 72

        lists = [b_f9, b_b8, p_f57, p_b72]  # beta 前9， beta 后8， pose 前57， pose 后72
        results = []

        def list_dot(b, p):
            res = itertools.product(lists[p], lists[b])
            l = list(res)
            results.extend(l)

        list_dot(0, 2)
        list_dot(1, 2)
        list_dot(0, 3)
        list_dot(1, 3)

        if self.end > 0:
            results = results[:self.end]

        results = results[self.start:]

        return results


if __name__ == '__main__':
    sp = SpecialMapper()
    sp.start = 100
    sp.end = 124
    sp.betas_list = np.linspace(1, 17, 17).astype('i')
    sp.poses_list = np.linspace(1, 129, 129).astype('i')

    res = sp.get_pairs()
    i = 0
    for r in res:
        print(r)
        i += 1
    print(i)





