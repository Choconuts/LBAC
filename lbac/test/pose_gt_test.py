from lbac.train.pose_gt import *
from com.sequence.sequence import *
from lbac.display.seq_show import *

pose_gt = None
pose_gt_dir = r'D:\Educate\CAD-CG\GitProjects\adj-33'
pose_gt_dir = r'D:\Educate\CAD-CG\GitProjects\LBAC\db\gt\pose\2'
# pose_gt_dir = r'D:\Educate\CAD-CG\GitProjects\adj-it'
pose_gt_dir = r'D:\Educate\CAD-CG\GitProjects\pose-2\1'


def load_data_of_seq(si):
    return load_json(join(pose_gt_dir, str5(int(si)) + '.json'))


def get_pose_gt():
    global pose_gt
    if pose_gt is None:
        pose_gt = PoseGroundTruth().load(pose_gt_dir)
    return pose_gt


def get_disp_seq(si):
    path = conf_path('tst_seqs/test_disp_' + str(si) + '.json','tst')
    if exists(path):
        return Sequence().load(path)
    print(get_pose_gt().index)
    disp_seq = Sequence(0.033, 'disp')
    disps = pose_gt.data[si]['disps']
    disp_seq.data = disps
    disp_seq.save(path)
    return disp_seq


def get_pose_seq(si):
    path = conf_path('tst_seqs_sm/test_pose_' + str(si) + '.json','tst')
    if exists(path):
        return Sequence().load(path)
    print(get_pose_gt().index)
    pose_seq = Sequence(0.033, 'pose')
    disps = pose_gt.data[si]['poses']
    pose_seq.data = disps
    pose_seq.save(path)
    return pose_seq


def show_disp_gt():
    data = load_data_of_seq(25)
    disp_seq = Sequence(0.033, "disp")
    pose_seq = Sequence(0.033, "pose")
    disp_seq.data = np.array(data['disps'])
    pose_seq.data = np.array(data['poses'])
    print(pose_seq.data)
    show_seqs(
        pose_disp_seq=disp_seq,
        # pose_seq=pose_seq,
        # show_body=True
    )


if __name__ == '__main__':
    show_disp_gt()
