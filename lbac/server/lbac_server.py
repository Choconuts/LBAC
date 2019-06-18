# encoding: utf-8
from com.memory.server import *
from com.memory.server import *
from com.mesh.mesh import *
from com.mesh.array_renderer import *
from tst.joint20.joints import *
from com.timer import Timer
from lbac.display.virtual_fitting import *

protocal = {
    'k': 21 * 3 * 4,
    'v': 7366 * 3 * 4,
    'f': 14496 * 3 * 4,
    'b': 10 * 4,
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

    smpl = SMPLModel(conf_path('smpl'))

    vf = VirtualFitting()

    def idle(m):
        m.seek(0)
        kinect_pos = m.read(size(['k']))
        kinect_pos = np.fromstring(kinect_pos, 'f').reshape((21, 3))

        axangle = kinect_pos[0]
        if not math.isnan(axangle[0]) and not np.linalg.norm(np.array(axangle)) == 0:
            ai, aj, ak = tr.euler.mat2euler(tr.axangles.axangle2mat(axangle, math.radians(np.linalg.norm(axangle))))
            root_rot = [ai, aj, ak]
        else:
            root_rot = [0, 0, 0]

        kinect_joints = kinect_pos[1:]

        pose = kinect_joints_to_smpl(kinect_joints, root_rot, g, smpl)

        print(pose)

        vf.pose = pose
        vf.recalculate_pose_displacement()

        verts = np.array(vf.cloth.vertices, 'f').tobytes()
        faces = np.array(vf.cloth.faces, 'i').tobytes()
        m.seek(size(['k']))

        m.write(verts)
        m.write(faces)
        m.flush()

    cn = create_shared_connect(size(['k', 'v', 'f', 'b']), 'lbac', 0.01)
    cn.idle.insert(0, idle)

    print('start')
    try:
        cn.start()
    except KeyboardInterrupt:
        print('stopped!')
        canvas.close()
        vf.close()
        return
    canvas.close()
    vf.close()


if __name__ == '__main__':
    serve()

