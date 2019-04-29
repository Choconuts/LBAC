from app2.smpl.smpl_np import SMPLModel
from app2.learning.ground_truth import BetaGroundTruth, PoseGroundTruth
from app2.geometry.closest_vertex import ClosestVertex
from app2.learning.mlp import MLP
from app2.display.shader.shaders import *
from app2.display.utils import *
from app2.configure import *
import numpy as np


def get_body(shape, pose=None):
    return smpl.set_params(beta=shape, pose=pose)


def get_cloth(shape):
    disp = mlp.predict([shape])[0].reshape((7366, 3))
    mesh = Mesh(beta_gt.template)
    mesh.vertices += disp
    return mesh


def apply_winkle(cloth, pose):
    dulp = []
    for i in range(20):
        dulp.append(pose)

    disp = np.array(mlp2.predict([np.array(dulp)])).reshape((20, 7366, 3))[10]
    disp = pose_gt.pose_disps[0][19]
    cloth.vertices += disp
    return cloth


def apply_pose(cloth, pose, rela):
    body_weights = smpl.weights
    smpl.set_params(pose)
    cloth_weights = np.zeros((len(cloth.vertices), 24))
    for i in range(len(cloth.vertices)):
        cloth_weights[i] = body_weights[rela[i]]
    cloth.vertices = PoseGroundTruth('../../app/data/beta_simulation/avg_smooth.obj', smpl).apply(cloth_weights, cloth.vertices)
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
            spread(i, bn * np.dot(vb - vc, bn) * .8)
        cloth.vertices[i] += cn * 0.002


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


if __name__ == '__main__':
    beta_gt = BetaGroundTruth().load('../../app/data/ground_truths/gt_files/beta_gt_4.json') \
        .load_template('../../app/data/beta_simulation/avg_smooth.obj')
    mlp = MLP().load('../learning/tst/beta_model/1')
    mlp2 = MLP(20 * 24 * 3, 20 * 7366 * 3).load('../learning/tst/pose_model1')
    smpl = SMPLModel(smpl_model_path)


    shape = beta_gt.betas[0]
    pose = [[0, 0, 0]]
    for i in range(23):
        pose.append([-0.1, -0.2, 0.1])
    pose_gt = PoseGroundTruth('../../app/data/beta_simulation/avg_smooth.obj', smpl).load(
        '../../app/data/ground_truths/gt_files/pose_gt_4.json')
    pose = pose_gt.pose_seqs[0][19]
    pose[0][0] = 0
    pose[0][1] = 0
    pose[0][2] = 0

    timer = Timer()
    cloth = get_cloth(shape)
    body = get_body(shape)

    # vertex_rela =  ClosestVertex().calc(cloth, body).save('shape/record_2.json')
    vertex_rela = ClosestVertex().load(vertex_relation_path)
    rela = vertex_rela.get_rela()

    apply_winkle(cloth, pose)

    body_posed = get_body(shape, pose)
    # apply_pose(cloth, pose, rela)
    rela = vertex_rela.update(cloth, body_posed).get_rela()
    # rela = vertex_rela.update(cloth, body).get_rela()
    # post_processing(cloth, body_posed, rela)
    # post_processing(cloth, body, rela, True)
    from app2.geometry.smooth import smooth
    # smooth_bounds(cloth, 10)
    smooth(cloth, 2)

    # saved_cloth = Mesh().load('../test/save_mesh.obj')
    cloth.update()
    cloth.save('shape/cloth.obj')
    draw(cloth, body)
    # draw(saved_cloth, body)

