from app.main.main import shape_model_path, pose_model_path, beta_gt, pose_gt, smpl, beta_ground_truth, pose_ground_truth, pose_sequences_dir
from app.smpl.smpl_np import SMPLModel
from app.learning.ground_truth import BetaGroundTruth, PoseGroundTruth
from app.geometry.closest_vertex import ClosestVertex
from app.learning.mlp import MLP
from app.learning.gru3 import GRU
from app.display.shader.shaders import *
from app.display.utils import *
from app.configure import *
from app.learning.regressor import *
import numpy as np


def get_body(shape, pose=None):
    return smpl.set_params(beta=shape, pose=pose)


def get_cloth(shape):
    disp = mlp.predict([shape])[0].reshape((7366, 3))
    mesh = Mesh(beta_gt.template)
    mesh.vertices += disp
    return mesh


def remove_rotation(pose):
    pose[0][0] = 0
    pose[0][1] = 0
    pose[0][2] = 0
    return pose


def apply_winkle(cloth, pose):
    disps = np.array(gru.predict_seq(pose_gt.pose_seqs[0]))[:, 4].reshape((-1, 7366, 3))
    cloth.vertices += disps[-50]
    return cloth


def apply_pose(cloth, pose, rela):
    body_weights = smpl.weights
    smpl.set_params(pose)
    cloth_weights = np.zeros((len(cloth.vertices), 24))
    for i in range(len(cloth.vertices)):
        cloth_weights[i] = body_weights[rela[i]]
    cloth.vertices = PoseGroundTruth('../../app/data/beta_simulation/avg_smooth.obj', smpl, pose_sequences_dir).apply(cloth_weights, cloth.vertices)
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
        bias = 0.002

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

    # to the close skin
    for i in range(len(cloth.vertices)):
        # d, n, f = get_closest_face(i)
        vc = cloth.vertices[i]
        vb = body.vertices[rela[i]]
        bn = body.normal[rela[i]]
        cn = cloth.normal[i]
        if np.dot(vb - vc, bn) > 0:
            spread(i, bn * np.dot(vb - vc, bn) * 0.5, 20)

    for r in range(20000):
        flag = True
        for i in range(len(cloth.vertices)):
            # d, n, f = get_closest_face(i)
            vc = cloth.vertices[i]
            vb = body.vertices[rela[i]]
            bn = body.normal[rela[i]]
            cn = cloth.normal[i]
            if np.dot(vb - vc, bn) > 0.001:
                spread(i, bn * np.dot(vb - vc, bn) * 1, 10)
                flag = False
        if flag:
            break

    cloth.update()
    for i in range(len(cloth.vertices)):
        # d, n, f = get_closest_face(i)
        vc = cloth.vertices[i]
        vb = body.vertices[rela[i]]
        cn = cloth.normal[i]
        bn = body.normal[rela[i]]
        cloth.vertices[i] += cn * 0.004 + bn * 0.004

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

    rot = 180

    def tes_draw():
        global rot
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        shader.draw()

        glPushMatrix()
        glScale(1.5, 1.5, 1.5)
        glRotatef(rot, 0, 1, 0)
        rot += 0.5
        rot %= 360
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


def create_sequence(cloth_temp):
    disps = np.array(gru.predict_seq(pose_gt.pose_seqs[0]))[:, 4].reshape((-1, 7366, 3))

    res = []
    for i in range(50, 60):
        cloth = Mesh(cloth_temp)
        cloth.vertices += disps[i]
        vertex_rela = ClosestVertex().load(vertex_relation_path)
        cloth = apply_pose(cloth, pose_gt.pose_seqs[0][i + gru.n_steps - 1], vertex_rela.get_rela())
        rela_posed = vertex_rela.update(cloth, body_posed).get_rela()
        post_processing(cloth, body_posed, rela_posed)

        from app.geometry.smooth import smooth, smooth_bounds
        smooth_bounds(cloth, 3)
        smooth(cloth, 2)
        res.append(cloth)

    return res


def create_body_sequence():
    poses = pose_gt.pose_seqs[0]

    res = []
    for i in range(gru.n_steps - 1 + 50, gru.n_steps - 1 + 60):
        body = smpl.set_params(pose=poses[i])
        res.append(body)

    return res



def animate(cloths, bodies):
    glutInit([])
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)  # 显示模式 双缓存
    glutInitWindowPosition(100, 100)  # 窗口位置
    glutInitWindowSize(800, 800)  # 窗口大小
    glutInitContextVersion(4, 3)  # 为了兼容
    glutInitContextProfile(GLUT_CORE_PROFILE)  # 为了兼容

    global vbo2, rot, shader
    verts_list = []

    for i in range(len(cloths)):
        verts = cloths[i].to_vertex_buffer()
        verts = VertexArray(verts).add_cols([1]).get()
        verts_list.append(verts)
        verts2 = bodies[i].to_vertex_buffer()
        verts2 = VertexArray(verts2).add_cols([0]).get()
        verts_list.append(verts2)
    num = len(verts_list[0]) + len(verts_list[1])
    verts_list = np.hstack(verts_list)

    rot = 180
    global frame, last_time
    frame = 0
    frame_rate = 20
    last_time = time.time()

    def tes_draw():
        global rot, frame, vbo2, last_time
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        shader.draw()

        glPushMatrix()
        glScale(1.5, 1.5, 1.5)
        glRotatef(rot, 0, 1, 0)
        # rot += 0.5
        rot %= 360
        glBindBuffer(GL_ARRAY_BUFFER, vbo2.id)
        glDrawArrays(GL_TRIANGLES, int(num / 7) * frame, int(num / 7))
        glPopMatrix()
        glDisableVertexAttribArray(0)  # 解析数据 例如一个矩阵里含有 位置 、颜色、多种信息
        glutSwapBuffers()
        if time.time() - last_time > 1 / frame_rate:
            frame += 1
            last_time = time.time()
        frame %= len(cloths)

    glutCreateWindow("test")  # 创建窗口
    glutDisplayFunc(tes_draw)  # 回调函数
    glutIdleFunc(tes_draw)  # 回调函数

    vbo2 = DynamicVBO()
    vbo2.bind(verts_list)
    shader = SimpleShader().color(0, [0.8, 0.8, 0.8]).color(1, [0.6, 0.2, 1]).color(2, [1, 0.2, 0.2])
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glutMainLoop()


if __name__ == '__main__':

    mlp = MLP().load(shape_model_path)
    gru = GRU(24 * 3, 7366 * 3, 5).load(pose_model_path)
    beta_gt.load(beta_ground_truth).load_template()
    pose_gt.load(pose_ground_truth)
    sr = ShapeRegressor(mlp)
    pr = PoseRegressor(gru)

    shape = beta_gt.betas[0]
    # shape = [0, -1, 2, 1, -1, 0, 0, 2, 0, 2]
    pose = pose_gt.pose_seqs[0][65]

    timer = Timer()
    cloth = get_cloth(shape)
    cloth = Mesh(beta_gt.template)
    cloth.vertices += sr.gen(beta_gt.betas[0])
    # cloth.vertices += pose_gt.pose_disps[0][-1]
    # cloth.save('cloth0.obj')
    body = get_body(shape)
    for i in range(5):
        pr.gen(shape, pose_gt.pose_seqs[0][60 + i])
    cloth.vertices += pr.gen(shape, pose_gt.pose_seqs[0][65])
    # apply_winkle(cloth, pose)

    vertex_rela = ClosestVertex().load(vertex_relation_path)
    rela = vertex_rela.get_rela()

    # post_processing(cloth, body, rela)

    body_posed = get_body(shape, pose)
    apply_pose(cloth, pose, rela)
    rela_posed = vertex_rela.update(cloth, body_posed).get_rela()
    # post_processing(cloth, body_posed, rela_posed)

    from app.geometry.smooth import smooth, smooth_bounds
    smooth_bounds(cloth, 3)
    smooth(cloth, 2)

    timer = Timer()
    from app.main.virtual_fitting import VirtualFitting
    vf = VirtualFitting(pose_gt, beta_gt, mlp, gru, vertex_rela)
    for i in range(1):
        vf.pose = pose_gt.pose_seqs[0][60 + i]
        vf.update()
    timer.tick('vf')
    draw(vf.cloth, vf.body)
    # animate(create_sequence(get_cloth(shape)), create_body_sequence())

