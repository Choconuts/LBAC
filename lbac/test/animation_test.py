from com.sequence.sequence import *
from com.mesh.mesh import *
from com.mesh.simple_display import *
from com.mesh.array_renderer import *
from lbac.train.pose_gt import PoseGroundTruth
from lbac.display.regressor import PoseRegressor
from com.learning.canvas import Canvas
from lbac.display.seq_show import *
from lbac.test.turbulent_test import predict_pose, model
from lbac.display.cloth_animation import *


def tst_amc_seq(seq_path):
    pose_seq = Sequence()
    pose_seq.load(seq_path)
    # show_pose_seq_joints(pose_seq)
    print(pose_seq.time_step)
    pose_seq.re_sampling(0.033).slice(4, 10)
    disp_seq = predict_pose(pose_seq, model)
    show_seqs(
        # pose_disp_seq=disp_seq,
        pose_seq=pose_seq
    )
    cloth_animation = ClothAnimation()
    # beta_gt = BetaGroundTruth().load(conf_path('beta_gt'))
    # cloth0 = beta_gt.template
    # cloth_animation.create_animate_mesh(cloth0, 'cloth', [0.7, 0.5, 0.9])
    # vert_seq = Sequence()
    # vert_seq.data = disp_seq.data + cloth0.vertices
    # cloth_animation.set_mesh_sequence(vert_seq, 'cloth')
    # cloth_animation.save(conf_path('tst_seqs/tst_anima.json','tst'))
    cloth_animation.load(conf_path('tst_seqs/tst_anima.json','tst'))

    c = cloth_animation.mesh_dict['cloth']
    for f in c['verts']:
        print(f)

    print(cloth_animation.frame_num)
    cloth_animation.show()





if __name__ == '__main__':
    model = r'D:\Educate\CAD-CG\GitProjects\s80-2'
    seq_path = conf_path('tst_seqs/amc_13_29.json', 'tst')
    # seq_path = '../../tst/tst_seqs/test_pose_18.json'

    tst_amc_seq(seq_path)