from lbac.train.pose_gt import PoseGroundTruth
from lbac.train.shape_gt import BetaGroundTruth
from com.mesh.mesh import Mesh
from com.mesh.array_renderer import *
from com.mesh.simple_display import *
from com.posture.smpl import SMPLModel
from com.path_helper import *


pose_gt = PoseGroundTruth(5, 0).load(conf_path('pose_gt'))
beta_gt = BetaGroundTruth().load(conf_path('beta_gt'))
smpl = SMPLModel(conf_path('smpl'))

cloth_template = Mesh(beta_gt.template)


def show_pose_gts(pose_gt=pose_gt):

    for si in pose_gt.data:
        disps, poses, beta = get_pose_gt(si)

        print(si, beta, len(poses))


def get_pose_gt(si):
    if type(si) == int:
        si = str(si)
    x = pose_gt.data[si]
    return x['disps'], x['poses'], x['beta']


def show_cloth_truth(disps, poses=None):
    global frame
    frame = 0

    if poses is not None and len(poses) != len(disps):
        print('length of inputs are different!')
        return

    display_mesh = Mesh(cloth_template)
    display_body = Mesh().from_vertices(smpl.verts, smpl.faces)

    def idle():
        global frame
        frame += 1
        frame %= len(disps)

        display_mesh.vertices = np.copy(cloth_template.vertices)
        display_mesh.vertices += disps[frame]
        display_mesh.update_normal_only()

        if poses is not None:
            smpl.set_params(poses[frame])
            display_body.from_vertices(smpl.verts, smpl.faces)

        glutPostRedisplay()

    def draw():
        msr.render()
        if poses is not None:
            bdy.render()

    msr = MeshRenderer(display_mesh, [0.6, 0.6, 0.6])
    bdy = MeshRenderer(display_body, [0.7, 0.7, 0.7])

    set_callbacks(idle)
    set_display(draw)
    run_glut()


def show_predict_result(beta, poses, show_body=False):
    from lbac.display.virtual_fitting import VirtualFitting
    vf = VirtualFitting()
    vf.beta = beta

    frame = 0

    def idle():
        global frame
        frame += 1
        frame %= len(poses)

        vf.pose = poses[frame]

        if not show_body:
            vf.update_cloth_only()
        else:
            vf.update()

        glutPostRedisplay()

    def draw():
        msr.render()
        if show_body:
            bdy.render()

    msr = MeshRenderer(vf.cloth, [0.6, 0.6, 0.6])
    bdy = MeshRenderer(vf.body, [0.7, 0.7, 0.7])

    set_callbacks(idle)
    set_display(draw)
    run_glut()



if __name__ == '__main__':
    disps, poses, beta = get_pose_gt(18)
    show_cloth_truth(disps[20:80])
    show_predict_result(beta, poses[20:80])