# encoding: utf-8
from com.memory.server import *
from com.mesh.mesh import *
from com.mesh.array_renderer import *
from tst.joint20.joints import *


protocal = {
    'k': 21 * 3 * 4,
    'v': 6890 * 3 * 4,
    'f': 13776 * 3 * 4
}

def size(keys):
    sum = 0
    for k in keys:
        sum += protocal[k]
    return sum


def serve():
    smpl = SMPLModel(conf_path('smpl'))

    from com.learning.canvas import Canvas
    from com.learning.mlp import Graph
    canvas = Canvas()
    g = Graph()
    canvas.load_graph(conf_path('temp/mlp/sklt/3'), g)

    def idle(m):
        m.seek(0)
        kinect_pos = m.read(size(['k']))
        kinect_pos = np.fromstring(kinect_pos, 'f').reshape((21, 3))
        root_rot = kinect_pos[0]
        for i in range(3):
            root_rot[i] = math.radians(root_rot[i])
        kinect_joints = kinect_pos[1:]
        # std = np.array([0, -0.295, 0])
        # tmp = kinect_joints[0] - std
        # for i in range(20):
        #     kinect_joints[i] -= tmp

        pose = kinect_joints_to_smpl(kinect_joints, root_rot, g)
        print(pose)
        smpl.set_params(pose)

        verts = np.array(smpl.verts, 'f').tobytes()
        faces = np.array(smpl.faces, 'i').tobytes()
        m.seek(size(['k']))

        m.write(verts)
        m.write(faces)
        m.flush()

    cn = create_shared_connect(size(['k', 'v', 'f']), 'pstm', 0.01)
    cn.idle.insert(0, idle)
    print('start')
    cn.start()
    canvas.close()


if __name__ == '__main__':
    serve()

