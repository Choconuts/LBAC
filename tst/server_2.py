# encoding: utf-8
from com.memory.server import *
from com.mesh.mesh import *
from com.protocal_1 import *

mesh = Mesh().load('tpl.obj')


vertices = mesh.vertices.astype('f')

info = ['v', 'f']


def serve():
    def idle(m):
        global vertices
        vertices += 0.001
        verts = vertices.tobytes()
        faces = np.array(mesh.faces, 'i').tobytes()
        m.seek(0)
        m.write(verts)
        m.write(faces)
        m.flush()

    cn = create_shared_connect(total_size(info), 'vf', 0.03)
    cn.idle.insert(0, idle)
    cn.start()


if __name__ == '__main__':
    serve()
        

