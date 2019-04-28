from app.learning.regressor import *
from app.smpl.smpl_np import smpl
from app.learning.ground_truths import beta_gt
from app.display.shader.shaders import *
from app.display.vbo import *
import numpy as np
from app.display.simple_display import VertexArray
from pyoctree import pyoctree as ot
from app.learning.ground_truths import PoseGroundTruth
import time
shape_regressor = ShapeRegressor()


def get_body(shape, pose=None):
    return smpl.set_params(beta=shape, pose=pose)


def get_cloth(shape):
    return shape_regressor.feed(shape).apply(beta_gt.template)


def apply_winkle(cloth, pose):
    return cloth


def apply_pose(cloth, body, pose, rela):
    body_weights = smpl.weights
    cloth_weights = np.zeros((len(cloth.vertices), 24))
    for i in range(len(cloth.vertices)):
        cloth_weights[i] = body_weights[rela[i]]
    cloth.vertices = PoseGroundTruth().apply(cloth_weights, cloth.vertices)
    cloth.update()
    return cloth

impact = []

def post_processing(cloth, body, rela, tst=False):
    import math

    def get_closest_face(i):
        vc = cloth.vertices[i]
        vb = body.vertices[rela[i]]
        face_map = body.vertex_face_map
        r0 = vc - vb
        min_d = 1000
        min_fn = [0, 0, 0]
        min_f = [0, 0, 0]
        for fi in face_map[rela[i]]:
            f = body.faces[fi]
            fn = cloth.face_norm(f)
            d = np.dot(fn, r0)
            r1 = body.vertices[f[0]] - vb
            r2 = body.vertices[f[1]] - vb
            r3 = body.vertices[f[2]] - vb
            eps = -0.0005
            if np.dot(r1, r0) < eps or np.dot(r2, r0) < eps or np.dot(r3, r0) < eps:
                continue
            if math.fabs(d) < min_d:
                min_d = math.fabs(d)
                min_fn = fn
                min_f = f
        return [np.dot(min_fn, r0), min_fn, min_f]

    global impact
    impact = []

    def spread(ci, vec, levels=40):
        past = {}
        bias = 0.06

        def get_vec(i, level):
            return vec * bias / (np.linalg.norm(cloth.vertices[i] - cloth.vertices[ci]) + bias)
            # return vec * level / tot_level

        def forward(i, level):
            if level == 0 or i in past:
                return
            past[i] = 1
            cloth.vertices[i] += get_vec(i, level)
            for ni in cloth.edges[i]:
                forward(ni, level - 1)

        forward(ci, levels)

    for i in range(len(cloth.vertices)):
        # d, n, f = get_closest_face(i)
        vc = cloth.vertices[i]
        vb = body.vertices[rela[i]]
        bn = body.normal[rela[i]]
        cn = cloth.normal[i]
        if np.dot(vb - vc, bn) > 0:
            spread(i, bn * np.dot(vb - vc, bn) * 1.2)
        cloth.vertices[i] += cn * 0.006


    return cloth


def draw(cloth, body):
    glutInit([])
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)  # 显示模式 双缓存
    glutInitWindowPosition(100, 100)  # 窗口位置
    glutInitWindowSize(800, 800)  # 窗口大小
    glutInitContextVersion(4, 3)  # 为了兼容
    glutInitContextProfile(GLUT_CORE_PROFILE)  # 为了兼容

    global vbo2, rot, shader
    verts = cloth.to_vertex_buffer()
    verts = VertexArray(verts).add_cols([1]).get()
    verts2 = body.to_vertex_buffer()
    verts2 = VertexArray(verts2).add_cols([0]).get()

    for i in impact:
        for fi in cloth.vertex_face_map[i]:
            verts[(fi * 3) * 7 + 6] = 2
            verts[(fi * 3 + 1) * 7 + 6] = 2
            verts[(fi * 3 + 2) * 7 + 6] = 2

    rot = 0

    def tes_draw():
        global rot
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        shader.draw()

        glPushMatrix()
        glScale(1.5, 1.5, 1.5)
        glRotatef(rot, 0, 1, 0)
        rot += 0.5
        glBindBuffer(GL_ARRAY_BUFFER, vbo2.id)
        glDrawArrays(GL_TRIANGLES, 0, int(vbo2.num / 7))
        glPopMatrix()
        glDisableVertexAttribArray(0)  # 解析数据 例如一个矩阵里含有 位置 、颜色、多种信息
        glutSwapBuffers()

    glutCreateWindow("test")  # 创建窗口
    glutDisplayFunc(tes_draw)  # 回调函数
    glutIdleFunc(tes_draw)  # 回调函数

    vbo2 = StaticVBO().bind(np.hstack((verts, verts2)))
    shader = SimpleShader().color(0, [0.8, 0.8, 0.8]).color(1, [0.6, 0.2, 1]).color(2, [1, 0.2, 0.2])
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glutMainLoop()


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

    timer.tick('hash')

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

    timer.tick('find')

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


def get_closest(cloth, body):
    rela = []
    for vc in cloth.vertices:
        min_d = 100
        min_v = -1
        for i in range(len(body.vertices)):
            d = np.linalg.norm(body.vertices[i] - vc)
            if d < min_d:
                min_d = d
                min_v = i
        rela.append(min_v)
    return rela


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


class VertexRelation:
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


global timer

if __name__ == '__main__':
    shape = beta_gt.betas[3]
    pose = [[0, 0, 0]]
    for i in range(23):
        pose.append([-0.1, -0.2, 0.1])
    pose = np.array(pose)
    cloth = get_cloth(shape)
    body = get_body(shape)
    timer = Timer(False)
    # vertex_rela =  VertexRelation().calc(cloth, body).save('shape/test_rela.json')
    vertex_rela = VertexRelation().load('shape/test_rela.json')
    rela = vertex_rela.get_rela()
    body_posed = get_body(shape, pose)
    # apply_pose(cloth, body, pose, rela)
    # rela = vertex_rela.update(cloth, body_posed).get_rela()
    rela = vertex_rela.update(cloth, body).get_rela()
    post_processing(cloth, body, rela)
    # post_processing(cloth, body, rela, True)
    # from app.geometry.smooth import smooth_bounds
    # smooth_bounds(cloth, 10)
    smooth(cloth, 2)

    # saved_cloth = Mesh().load('../test/save_mesh.obj')
    cloth.update()
    cloth.save('shape/cloth.obj')
    draw(cloth, body)
    # draw(saved_cloth, body)

