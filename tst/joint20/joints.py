from com.posture.smpl import *
from com.mesh.simple_display import *
from com.path_helper import *
import transforms3d as tr
import math
from tst.joint20.skeleton_nn import *
from com.timer import Timer


kinect_to_smpl = {
    0: 0,
    1: 6,
    2: 12,
    3: 15,
    4: 16,
    5: 18,
    6: 20,
    7: 22,
    8: 17,
    9: 19,
    10: 21,
    11: 23,
    12: 1,
    13: 4,
    14: 7,
    15: 10,
    16: 2,
    17: 5,
    18: 8,
    19: 11
}


def show_20_joints():
    smpl = SMPLModel(conf_path('smpl'))
    joints = smpl.J

    # joints = load_json('sklt.json')
    # rot_0 = joints[0]
    # joints = joints[1:]

    def d():
        glColor3f(1, 0.5, 0)
        glPointSize(5)
        glBegin(GL_POINTS)
        for j in joints:
            glVertex3d(j[0], j[1], j[2])
        glEnd()

    set_display(d)
    run_glut()


def save_20_joints():
    smpl = SMPLModel(conf_path('smpl'))
    joints = smpl.J

    in_file = conf_path('temp/128_r.json')
    seqs = load_json(in_file)

    # pose = seqs[33][25]
    #
    # smpl.set_params(np.array(pose))
    # joints = smpl.J
    # weights = np.eye(24)
    # joints = apply(smpl, weights, np.array(joints))


    pose = np.zeros((24, 3))
    # pose[1] = [0, 0, 0.2]

    smpl.set_params(np.array(pose))
    joints = smpl.J
    weights = np.eye(24)
    joints = apply(smpl, weights, np.array(joints))

    out = [[0, 0, 0]]
    for i in kinect_to_smpl:
        out.append(joints[kinect_to_smpl[i]].tolist())

    print(len(out))
    save_json(out, 'sklt2.json')


def show_joints(joints):
    def d():
        draw_axis()
        glColor3f(1, 0.5, 0)
        glPointSize(5)
        glBegin(GL_POINTS)
        for j in joints:
            glVertex3d(j[0], j[1], j[2])
        glEnd()

    set_display(d)
    run_glut()


def compare_joints(joints1, joints2):
    def d():
        draw_axis()
        glColor3f(1, 0.3, 0)
        glPointSize(5)
        glBegin(GL_POINTS)
        for j in joints1:
            glVertex3d(j[0], j[1], j[2])
        glEnd()

        glColor3f(0, 0.3, 1)
        glPointSize(5)
        glBegin(GL_POINTS)
        for j in joints2:
            glVertex3d(j[0], j[1], j[2])
        glEnd()

    set_display(d)
    run_glut()


def kinect_joints_to_smpl(kinect_joints, root_rot, graph, smpl):
    timer = Timer()
    if smpl is None:
        smpl = SMPLModel(conf_path('smpl'))
    smpl_joints = np.zeros((24, 3))

    kinect_joints[0] += 0.3 * (kinect_joints[1] - kinect_joints[0])

    for i in kinect_to_smpl:
        smpl_joints[kinect_to_smpl[i]] = kinect_joints[i]

    smpl_joints = process_joints(smpl_joints, root_rot)

    timer.tick('preprocess')

    # 将获得的关节点表示到右手坐标系下
    root_mat = tr.euler.euler2mat(*root_rot)
    root_mat[0, 2] *= -1
    root_mat[1, 2] *= -1
    root_mat[2, 0] *= -1
    root_mat[2, 1] *= -1
    root_rot = tr.euler.mat2euler(root_mat)
    smpl_joints[:, 2] *= -1

    x = []
    for i in [6, 12, 16, 17]:
        x.append(smpl_joints[i].tolist())
    y = graph.predict(None, [np.array([x]).reshape(-1, 12)]).reshape(-1, 4, 3)[0]
    mp = [3, 9, 13, 14]
    for i in range(4):
        smpl_joints[mp[i]] = y[i]

    timer.tick('mlp')

    pose = joints_to_smpl(smpl, smpl_joints, root_rot)

    timer.tick('pose')

    def show():
        smpl.set_params(pose)
        joints = smpl.J
        weights = np.eye(24)
        joints = apply(smpl, weights, np.array(joints))

        def d():
            glColor3f(1, 0, 0)
            glPointSize(5)
            glBegin(GL_POINTS)
            for j in joints:
                j = np.matmul(np.linalg.inv(tr.euler.euler2mat(*root_rot)), j)
                glVertex3d(j[0], j[1], j[2])
            glEnd()

        set_display(d)
        run_glut()

    # smpl.set_params(pose)
    # joints = smpl.J
    # weights = np.eye(24)
    # joints = apply(smpl, weights, np.array(joints))
    # compare_joints(smpl_joints, process_joints(joints, [0, 0, 0]))

    return pose


def joints_to_smpl(smpl, smpl_joints, root_rot):

    def norm(vec):
        n = np.linalg.norm(vec)
        if n == 0:
            return None
        return vec / n

    global_rot_mat = tr.euler.euler2mat(*root_rot)
    # for i in range(len(smpl_joints)):
    #     smpl_joints[i] = np.matmul(np.linalg.inv(global_rot_mat), smpl_joints[i])

    joints_mat = dict()
    not_reliable = []

    for i in range(1, 24):
        vec0 = norm(smpl.J[i] - smpl.J[smpl.parent[i]])
        vec1 = norm(smpl_joints[i] - smpl_joints[smpl.parent[i]])
        if vec0 is None or vec1 is None:
            return np.zeros((24, 3))
        if np.linalg.norm(vec0 - vec1) < 1e-8:
            mat = np.eye(3)
        else:
            # if np.linalg.norm(vec0 - vec1) > 0.01:
            #     print(i, smpl.parent[i], vec0, vec1)
            mat = tr.axangles.axangle2mat(np.cross(vec0, vec1), math.acos(
                min(np.dot(vec0, vec1) / (np.linalg.norm(vec0) * np.linalg.norm(vec1)) , 1)
            ))
        if smpl.parent[i] not in joints_mat:
            joints_mat[smpl.parent[i]] = []
        joints_mat[smpl.parent[i]].append(mat)
        if i in [3, 9, 13, 14]:
            not_reliable.append([smpl.parent[i], len(joints_mat[smpl.parent[i]]) - 1])

    def mat2ax(mat):
        ax, rad = tr.axangles.mat2axangle(mat)
        ax = ax[0:3]
        return ax / np.linalg.norm(ax) * rad

    # pose = [mat2ax(mats[0])]
    # for i in range(1, 24):
    #     pose.append(mat2ax(np.matmul((mats[smpl.parent[i]]), mats[i])))
    #
    # print(pose)

    pose = np.zeros((24, 3))
    joints_mat[-1] = [np.eye(3)]
    smpl.parent[0] = -1

    # reduce mean
    for j in joints_mat:
        mts = joints_mat[j]
        if len(mts) > 1:
            s = []
            i = -1
            for m in mts:
                i += 1
                if [j, i] in not_reliable:
                    continue
                qua = tr.quaternions.mat2quat(m)
                s.append(qua)
            if len(s) >= 1:
                s = np.array(s)
                qua = np.mean(s, 0)
                joints_mat[j] = [tr.quaternions.quat2mat(qua)]
            else:
                joints_mat[j] = [np.eye(3)]

    for j in joints_mat:
        if j == -1:
            break
        mts = joints_mat[j]
        mat0 = np.matmul(np.linalg.inv(joints_mat[smpl.parent[j]][0]), mts[0])

        pose[j] = mat2ax(mat0)

    pose = np.array(pose)
    ax, angle = tr.axangles.mat2axangle(global_rot_mat)
    pose[0] = ax * angle

    return pose


def gen_smpl_joints():
    smpl = SMPLModel(conf_path('smpl'))
    joints = smpl.J

    pose = np.zeros((24, 3))
    pose[1] = [0, 0, 0.2]

    smpl.set_params(np.array(pose))
    joints = smpl.J
    weights = np.eye(24)
    joints = apply(smpl, weights, np.array(joints))

    return joints


def test_process_joints():
    smpl = SMPLModel(conf_path('smpl'))
    pose = np.zeros((24, 3))
    pose[0] = (0, 0, 2)
    pose[1:] += 0
    smpl.set_params(pose)
    joints = smpl.J
    joints = apply(smpl, np.eye(24), joints)

    joints = process_joints(joints, pose[0])
    show_joints(joints)


if __name__ == '__main__':
    # show_20_joints()
    # save_20_joints()


    sklt = load_json('sklt.json')

    from com.learning.canvas import Canvas
    from com.learning.mlp import Graph
    canvas = Canvas()
    g = Graph()
    canvas.load_graph(conf_path('temp/mlp/sklt/3'), g)

    kinect_joints_to_smpl(sklt[1:], [0, 0, 0], g)

    canvas.close()

    # test_process_joints()
