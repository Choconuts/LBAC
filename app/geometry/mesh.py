import numpy as np
import re
from app.display.objects import OBJ
import copy


class Mesh:

    def __init__(self, another_mesh=None):
        if another_mesh is not None:
            self.vertices = np.copy(another_mesh.vertices)
            self.faces.extend(another_mesh.faces)
            self.edges.update(another_mesh.edges)
            self.bounds.update(another_mesh.bounds)
        else:
            self.vertices = np.array([])
            self.faces = []
            self.edges = dict()
            self.bounds = dict()

    def load(self, obj_file_path):
        vertices = []
        bound_edges = dict()
        self.faces= []
        g = re.search('[a-zA-Z0-9$\\- ].obj$', obj_file_path)
        if g is None:
            return
        g = g.span(0)
        if obj_file_path is not None:
            obj = OBJ(obj_file_path[0:g[0]], obj_file_path[g[0]:g[1]])
            vertices.extend(obj.vertices)
            for f in obj.faces:
                face = [f[0][0] - 1, f[0][1] - 1, f[0][2] - 1]
                self.faces.append(face)
                for i in range(3):
                    edge = tuple([face[i], face[(i + 1) % 3]])
                    self.add_edge(edge[0], edge[1])
                    self.add_edge(edge[1], edge[0])

                    # record bound edges
                    if edge not in bound_edges:
                        bound_edges[(edge[1], edge[0])] = 1
                    else:
                        bound_edges.pop(edge)
            for e in bound_edges:
                if e[0] not in self.bounds:
                    self.bounds[e[0]] = 1
                if e[1] not in self.bounds:
                    self.bounds[e[1]] = 1
        self.vertices = np.array(vertices)
        return self

    def from_vertices(self, vertices, faces):
        bound_edges = dict()
        for face in faces:
            self.faces.append(np.copy(face).tolist())
            for i in range(3):
                edge = tuple([face[i], face[(i + 1) % 3]])
                self.add_edge(edge[0], edge[1])
                self.add_edge(edge[1], edge[0])

                # record bound edges
                if edge not in bound_edges:
                    bound_edges[(edge[1], edge[0])] = 1
                else:
                    bound_edges.pop(edge)
        for e in bound_edges:
            if e[0] not in self.bounds:
                self.bounds[e[0]] = 1
            if e[1] not in self.bounds:
                self.bounds[e[1]] = 1
        self.vertices = np.array(vertices)
        return self

    def add_edge(self, a, b):
        if a not in self.edges:
            self.edges[a] = []
        if b not in self.edges[a]:
            self.edges[a].append(b)

    def save(self, path):
        with open(path, 'w') as fp:
            for v in self.vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in np.array(self.faces) + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        return self

    def to_vertex_buffer(self):
        buffer = []
        for face in self.faces:
            triangle = []
            for v in face:
                triangle.append(self.vertices[v])
            buffer.append(triangle)
        return np.array(buffer)
