from app.display.objects import OBJ
import numpy as np
from app.geometry.mesh import Mesh


def find_neighbors(edges, v, r, neighbors):
    if r <= 0:
        return
    for n in edges[v]:
        if n in neighbors:
            continue
        neighbors.append(n)
        find_neighbors(edges, n, r - 1, neighbors)


def avg_neighbors(vertices, edges, v, r):
    neighbors = [v]
    find_neighbors(edges, v, r, neighbors)
    s = 0
    ws = 0
    for n in neighbors:
        if n == v:
            continue
        w = 1 / np.linalg.norm(vertices[n] - vertices[v])
        s += w * vertices[n]
        ws += w

    s /= ws
    return s


def smooth(mesh, times=1, level=1):
    """

    :param mesh: 需要平滑的网格
    :param times: 平滑迭代的次数, 1~2次可以去噪声，5~10次可以去坑洞，40~50次可以去除布褶，建议不要超过60次
    :param level: 平滑卷积的范围，通常是1不要修改
    :return: 平滑网格
    """
    for i in range(times):
        new_vertices = []
        new_vertices.extend(mesh.vertices)
        for v in mesh.edges:
            neighbors = [v]
            find_neighbors(mesh.edges, v, level, neighbors)
            s = 0
            ws = 0
            for n in neighbors:
                bias = 1
                if n in mesh.bounds:
                    bias = 10
                    if n == v:
                        bias = 50
                w = 1 / (np.dot(mesh.vertices[n] - mesh.vertices[v], mesh.vertices[n] - mesh.vertices[v]) + 0.002) * bias
                s += w * mesh.vertices[n]
                ws += w
            if ws > 0:
                new_vertices[v] = s / ws
        mesh.vertices = np.array(new_vertices)
    return mesh


if __name__ == '__main__':
    m = Mesh().load('../data/beta_simulation/result/12.obj')
    smooth(m, 50)
    m.save('../test/save_mesh.obj')
