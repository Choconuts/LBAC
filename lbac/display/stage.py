from lbac.display.virtual_fitting import *
from com.mesh.array_renderer import *
from com.mesh.simple_display import *

vf = None


pose_gt = PoseGroundTruth(5, 0).load(conf_path('pose_gt'))


def main():
    global vf
    if vf is None:
        vf = VirtualFitting()

    # vf.beta = np.random.random((10)) * 2 - 1
    # vf.pose = np.random.random((24, 3)) * 0.1

    # vf.update_cloth_only()
    # vf.update_cloth_only()
    # vf.update_cloth_only()


    def draw():
        msr.render()

    msr = MeshRenderer(vf.cloth, [0.8, 0.8, 0.8])
    set_display(draw)
    set_init(init_array_renderer)
    run_glut()

    vf.close()


def view_mesh(mesh):
    global frame

    frame = 0

    def idle():
        pass

    def draw():
        msr.render()

    msr = MeshRenderer(mesh, [0.8, 0.8, 0.8])
    set_display(draw)
    set_callbacks(idle)
    set_init(init_array_renderer)
    run_glut()


def view_meshes(meshes):
    global frame

    frame = 0

    def idle():
        pass

    def draw():
        for msr in msrs:
            msr.render()

    msrs = []
    for mesh in meshes:
        msrs.append(MeshRenderer(mesh, [0.8, 0.8, 0.8]))
    set_display(draw)
    set_callbacks(idle)
    set_init(init_array_renderer)
    run_glut()


def tst_train_data():
    global vf, frame
    if vf is None:
        vf = VirtualFitting()

    vf.beta = np.hstack((np.array(pose_gt.data['18']['beta']), np.zeros(6)))
    vf.pose = np.array(pose_gt.data['18']['poses'][0]) + 0.05

    vf.update_cloth_only()
    # vf.update_cloth_only()
    # vf.update_cloth_only()

    frame = 0

    def idle():
        global frame
        frame += 1
        frame %= 45
        vf.pose = np.array(pose_gt.data['18']['poses'][frame]) + 0.05
        vf.update_cloth_only()

        glutPostRedisplay()

    def draw():
        msr.render()

    msr = MeshRenderer(vf.cloth, [0.8, 0.8, 0.8])
    set_display(draw)
    set_callbacks(idle)
    set_init(init_array_renderer)
    run_glut()

    vf.close()


def check_gt():
    print(len(pose_gt.data))

    vf = VirtualFitting()

    vf.cloth.vertices += pose_gt.data['18']['disps'][40]
    vf.smpl.set_params(pose=np.array(pose_gt.data['18']['poses'][40]))
    vf.cloth.vertices = apply_pose(vf.smpl, vf.cloth_weights, vf.cloth.vertices)

    main()
    vf.close()


if __name__ == '__main__':
    tst_train_data()
