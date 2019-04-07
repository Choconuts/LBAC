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
    for i in range(times):
        new_vertices = []
        new_vertices.extend(mesh.vertices)
        for v in mesh.edges:
            if v in mesh.bounds:
                continue
            new_vertices[v] = avg_neighbors(mesh.vertices, mesh.edges, v, level)
        mesh.vertices = np.array(new_vertices)
    return mesh


if __name__ == '__main__':
    m = Mesh().load('../test/result/12.obj')
    smooth(m, 10)
    m.save('../test/save_mesh.obj')
