import numpy as np
import time


def get_closest_points(cloth, body):
    """
    :param cloth: 7366 vertices
    :param body: 6890 vertices
    :return: 每个衣服顶点最近的人体顶点标号的列表
    """
    # hash the body
    import math
    class LinearFunc:
        """
        先用二次把
        """
        def __init__(self, thrs, ks): # [?, 0.5, 0.7], [1.2, 1, ?]
            import copy
            self.thrs = copy.deepcopy(thrs)
            self.thrs.insert(0, 0)
            self.ks = copy.deepcopy(ks)
            k = 1
            for i in range(len(ks)):
                k -= ks[i] * (self.thrs[i + 1] - self.thrs[i])
            self.ks.append(k / (1 - self.thrs[len(ks)]))
            self.bs = [0]
            for i in range(len(ks)):
                self.bs.append(self.bs[len(self.bs) - 1] + ks[i] * (self.thrs[i + 1] - self.thrs[i]))

        def fin(self, x):
            y = 0
            for i in range(len(self.thrs) - 1):
                y += (self.ks[i] * (x - self.thrs[i]) + self.bs[i]) * (self.thrs[i] < x < self.thrs[i + 1])
            return x

        def fout(self, x):
            return x ** 2

    lf = LinearFunc([0.2, 0.5], [2, 1.5])
    resolution = 80

    min_x = np.min(body.vertices)
    max_x = np.max(body.vertices)

    def my_hash(x, step):
        x = (x - min_x) / (max_x - min_x + 0.0001)
        y = int(lf.fin(x) / step)
        assert y >= 0
        return y

    # 建立多层哈希
    resolu = resolution
    resolutions = []
    hash_tables = []
    lists_map = []
    while resolu > 0:
        resolutions.append(resolu)
        step = 1 / resolu
        hash_table = np.zeros((resolu, resolu, resolu), np.int)
        lists = [[]]
        i = 1
        vi = 0
        for v in body.vertices:
            v0 = my_hash(v[0], step)
            v1 = my_hash(v[1], step)
            v2 = my_hash(v[2], step)
            if hash_table[v0][v1][v2] == 0:
                hash_table[v0][v1][v2] = i
                lists.append([vi])
                i += 1
            else:
                lists[hash_table[v0][v1][v2]].append(vi)
            vi += 1


        lists_map.append(lists)
        hash_tables.append(hash_table)
        resolu = int(resolu / 5)

    # timer.tick('hash')

    res = []
    for v in cloth.vertices:
        search_list = []
        layer = 0
        while True:
            step = 1 / resolutions[layer]
            v0 = my_hash(v[0], step)
            v1 = my_hash(v[1], step)
            v2 = my_hash(v[2], step)
            if hash_tables[layer][v0][v1][v2] == 0:
                layer += 1
                continue
            if layer < len(resolutions) - 1:
                layer += 1
            step = 1 / resolutions[layer]
            v0 = my_hash(v[0], step)
            v1 = my_hash(v[1], step)
            v2 = my_hash(v[2], step)
            search_list.extend(lists_map[layer][hash_tables[layer][v0][v1][v2]])
            break
        closest_v = -1
        closest_dist = 100
        for p in search_list:
            d = np.linalg.norm(v - body.vertices[p])
            if d < closest_dist:
                closest_dist = d
                closest_v = p
        res.append(closest_v)

    # timer.tick('find')

    def travel():
        for i in range(len(cloth.vertices)):
            now = res[i]
            vc = cloth.vertices[i]
            while True:
                min_d = np.linalg.norm(vc - body.vertices[now])
                min_v = -1
                for v in body.edges[res[i]]:
                    d = np.linalg.norm(vc - body.vertices[v])
                    if d < min_d:
                        min_d = d
                        min_v = v
                if min_v < 0:
                    break
                now = min_v
            res[i] = now
    travel()

    return res


class Timer:
    def __init__(self, prt=True):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.print = prt

    def tick(self, msg=''):
        res = time.time() - self.last_time
        if self.print:
            print(msg, res)
        self.last_time = time.time()
        return res

    def tock(self, msg=''):
        if self.print:
            print(msg, time.time() - self.last_time)
        return time.time() - self.last_time


class ClosestVertex:
    def __init__(self):
        self.relation = []

    def get_rela(self):
        return self.relation

    def save(self, file):
        import json
        with open(file, 'w') as fp:
            json.dump(self.relation, fp)
        return self

    def load(self, file):
        import json
        with open(file, 'r') as fp:
            self.relation = json.load(fp)
        return self

    def calc(self, cloth, body):
        self.relation = get_closest_points(cloth, body)
        return self

    def update(self, cloth, body):
        for i in range(len(cloth.vertices)):
            now = self.relation[i]
            vc = cloth.vertices[i]
            while True:
                min_d = np.linalg.norm(vc - body.vertices[now])
                min_v = -1
                for v in body.edges[self.relation[i]]:
                    d = np.linalg.norm(vc - body.vertices[v])
                    if d < min_d:
                        min_d = d
                        min_v = v
                if min_v < 0:
                    break
                now = min_v
            self.relation[i] = now
        return self


if __name__ == '__main__':
    pass

