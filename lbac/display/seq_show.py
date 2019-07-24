from com.sequence.sequence import *
from com.mesh.array_renderer import *
from com.mesh.simple_display import *
from com.mesh.mesh import *
from com.posture.smpl import *

smpl = None
frame_counter = 0
time_counter = 0
last_draw_frame = -1
real_timing = True
mesh_color = [0.5, 0.8, 1]


def read_smpl(path=None):
    if path is None:
        path = conf_path('smpl')
    global smpl
    if smpl is None:
        smpl = SMPLModel(path)
    return smpl


def show_sequence_mesh(seq: Sequence, mesh, mesh_changer):
    global frame_counter, time_counter, last_draw_frame
    frame_counter = 0
    tot_frame = seq.get_frame_num()
    time_counter = 0
    tot_time = seq.get_total_time()
    print('total time: %f' % tot_time)
    start_time = time.time()
    last_draw_frame = -1

    meshes = []
    for i in range(tot_frame):
        print(i)
        mesh_changer(mesh, seq.data[i])
        meshes.append(Mesh(mesh))

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
                # mesh_changer(mesh, seq.data[next_frame])
                msr.mesh = meshes[next_frame]
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


def show_pose_seq(pose_seq):
    smpl = read_smpl()
    mesh = Mesh().from_vertices(smpl.verts, smpl.faces)
    def mesh_changer(m: Mesh, shot):
        smpl.set_params(pose=np.array(shot).reshape(24, 3))
        m.from_vertices(smpl.verts, smpl.faces)

    show_sequence_mesh(pose_seq, mesh, mesh_changer)


def show_seqs(pose_seq, beta_disp, pose_disp_seq):
    pass


if __name__ == '__main__':
    path = '../../tst/test_show_pose_seq.json'
    pose_seq = Sequence(0.033, 'pose').load(path)
    show_pose_seq_joints(pose_seq)
