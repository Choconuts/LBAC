from com.sequence.sequence import *
from com.mesh.array_renderer import *
from com.mesh.simple_display import *
from com.mesh.mesh import *
from com.posture.smpl import *
from lbac.train.shape_gt import *
from com.mesh.closest_vertex import *


smpl = None
frame_counter = 0
time_counter = 0
last_draw_frame = -1
real_timing = True
mesh_color = [0.5, 0.8, 1]
beta_gt = None
relation = None
cloth_weights = None

def read_smpl(path=None):
    if path is None:
        path = conf_path('smpl')
    global smpl
    if smpl is None:
        smpl = SMPLModel(path)
    return smpl


def read_vertex_relation(path=None):
    if path is None:
        path = conf_path('vert_rela')
    global relation, cloth_weights, beta_gt
    if relation is None:
        relation = ClosestVertex().load(path)
    if beta_gt is None:
        beta_gt = BetaGroundTruth().load(conf_path('beta_gt'))
    if cloth_weights is None:
        cloth_weights = np.zeros((len(beta_gt.template.vertices), 24))
        rela = relation.get_rela()
        for i in range(len(beta_gt.template.vertices)):
            cloth_weights[i] = read_smpl().weights[rela[i]]


def show_sequence_mesh(seq: Sequence, mesh, mesh_changer):
    global frame_counter, time_counter, last_draw_frame
    frame_counter = 0
    tot_frame = seq.get_frame_num()
    time_counter = 0
    tot_time = seq.get_total_time()
    print('total time: %f' % tot_time)
    start_time = time.time()
    last_draw_frame = -1

    from com.timer import Timer

    # meshes = []
    # for i in range(tot_frame):
    #     print(i)
    #     tm = Timer()
    #     m = Mesh(mesh)
    #     tm.tick()
    #     mesh_changer(m, seq.data[i])
    #     tm.tick()
    #     meshes.append(m)

    def idle():
        global frame_counter, time_counter
        frame_counter += 1
        frame_counter %= tot_frame
        time_counter = time.time() - start_time
        time_counter %= tot_time
        global last_draw_frame
        if real_timing:
            next_frame = int(math.floor(time_counter / seq.time_step))
            if next_frame != last_draw_frame:
                mesh_changer(msr.mesh, seq.data[next_frame])
                # msr.mesh = meshes[next_frame]
                last_draw_frame = next_frame
                glutPostRedisplay()
        else:
            mesh_changer(mesh, seq.data[frame_counter])
            glutPostRedisplay()

    def draw():
        msr.render()

    def init():
        init_array_renderer()

    msr = MeshRenderer(mesh, mesh_color)
    set_init(init)
    set_callbacks(idle)
    set_display(draw)
    run_glut()


def show_pose_seq_joints(pose_seq):
    smpl = read_smpl()

    global frame_counter, time_counter, last_draw_frame
    frame_counter = 0
    tot_frame = pose_seq.get_frame_num()
    time_counter = 0
    tot_time = pose_seq.get_total_time()
    print('total time: %f' % tot_time)
    start_time = time.time()
    last_draw_frame = -1

    joints = []
    for i in range(tot_frame):
        smpl.set_params(pose=pose_seq.data[i].reshape(24, 3))
        j = np.copy(smpl.J)
        weights = np.eye(24)
        j = apply(smpl, weights, np.array(j))
        joints.append(j)

    def idle():
        global frame_counter, time_counter
        frame_counter += 1
        frame_counter %= tot_frame
        time_counter = time.time() - start_time
        time_counter %= tot_time
        global last_draw_frame
        if real_timing:
            next_frame = int(math.floor(time_counter / pose_seq.time_step))
            if next_frame != last_draw_frame:
                last_draw_frame = next_frame
                glutPostRedisplay()
        else:
            last_draw_frame = frame_counter
            glutPostRedisplay()

    def draw():
        glColor3f(0.8, 0.5, 0)
        glPointSize(5)
        glBegin(GL_POINTS)
        for p in joints[last_draw_frame]:
            glVertex3d(p[0], p[1], p[2])
        glEnd()

    def init():
        init_array_renderer()

    set_init(init)
    set_callbacks(idle)
    set_display(draw)
    run_glut()


def show_multi_pose_seq_joints(list_of_pose_seq):
    smpl = read_smpl()

    global time_counter, last_draw_frames
    time_counter = 0
    tms = []
    for ps in list_of_pose_seq:
        tms.append(ps.get_total_time())
    tot_time = min(tms)
    print('total time: %f' % tot_time)
    start_time = time.time()
    last_draw_frames = []
    joints_list = []
    color_list = [[0.8, 0.8, 0], [0, 0.8, 0.8], [0.8, 0, 0.8], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]]

    for ps in list_of_pose_seq:
        last_draw_frames.append(-1)
        joints_list.append([])
        for i in range(ps.get_frame_num()):
            smpl.set_params(pose=ps.data[i].reshape(24, 3))
            j = np.copy(smpl.J)
            weights = np.eye(24)
            j = apply(smpl, weights, np.array(j))
            joints_list[-1].append(j)

    def idle():
        global time_counter
        time_counter = time.time() - start_time
        time_counter %= tot_time
        global last_draw_frames
        flag = False
        for pi in range(len(list_of_pose_seq)):
            next_frame = int(math.floor(time_counter / list_of_pose_seq[pi].time_step))
            if next_frame != last_draw_frames[pi]:
                last_draw_frames[pi] = next_frame
                flag = True
        if flag:
            glutPostRedisplay()

    def draw():
        for i in range(len(list_of_pose_seq)):
            glColor3f(*color_list[i])
            glPointSize(5)
            glBegin(GL_POINTS)
            for p in joints_list[i][last_draw_frames[i]]:
                glVertex3d(p[0], p[1], p[2])
            glEnd()

    def init():
        init_array_renderer()

    set_init(init)
    set_callbacks(idle)
    set_display(draw)
    run_glut()


def show_pose_seq(pose_seq):
    smpl = read_smpl()
    mesh = Mesh().from_vertices(smpl.verts, smpl.faces)
    def mesh_changer(m: Mesh, shot):
        smpl.set_params(pose=np.array(shot).reshape(24, 3))
        m.from_vertices(smpl.verts, smpl.faces)

    show_sequence_mesh(pose_seq, mesh, mesh_changer)


def show_seqs(beta: np.ndarray = None, pose_seq: Sequence = None, beta_disp: np.ndarray = None, pose_disp_seq: Sequence = None):
    global beta_gt
    if beta_gt is None:
        beta_gt = BetaGroundTruth().load(conf_path('beta_gt'))

    if beta is None:
        beta = np.zeros(10)
    elif len(beta) == 4:
        beta = np.hstack(beta, np.zeros(6))
    verts_temp = np.copy(beta_gt.template.vertices)
    frame_num = 1
    if pose_seq:
        frame_num = pose_seq.get_frame_num()
    elif pose_disp_seq:
        frame_num = pose_disp_seq.get_frame_num()

    index_seq = Sequence()
    index_seq.data = np.linspace(0, frame_num - 1, frame_num)

    def mesh_process(mesh, i):
        i = int(i)
        # pose disp
        mesh.vertices = np.copy(verts_temp)
        update_normal_flag = False
        if pose_disp_seq:
            mesh.vertices += pose_disp_seq.data[i]
        if beta_disp:
            mesh.vertices += beta_disp
        if pose_seq:
            update_normal_flag = True
            read_smpl()
            read_vertex_relation()
            smpl.set_params(pose=pose_seq.data[i], beta=beta, mat_only=True)
            mesh.vertices = apply(smpl, cloth_weights, mesh.vertices)
        if update_normal_flag:
            mesh.update_normal_only()
        return mesh

    show_sequence_mesh(index_seq, Mesh(beta_gt.template), mesh_process)


def show_disps(pose_disp_seq):
    global beta_gt
    if beta_gt is None:
        beta_gt = BetaGroundTruth().load(conf_path('beta_gt'))

    verts_temp = np.copy(beta_gt.template.vertices)

    def get_disped_mesh(mesh, shot):
        mesh.vertices = verts_temp + shot.reshape(-1, 3)
    show_sequence_mesh(pose_disp_seq, Mesh(beta_gt.template), get_disped_mesh)


if __name__ == '__main__':
    path = '../../tst/test_show_pose_seq.json'
    pose_seq = Sequence(0.033, 'pose').load(path)
    show_pose_seq_joints(pose_seq)
