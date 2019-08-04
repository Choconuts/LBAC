from lbac.sequence.pose_translator import *
from lbac.sequence.reader import *
from com.sequence.sequence import *
from lbac.display.seq_show import *
from tst.workspace.seq_cut_0 import process_seqs


def get_pose_sequence():
    # amc_translator = AmcTranslator()
    # amc_translator.meta = {
    #     "dir": "raw/mocap/1",
    #     "gen": {"13": [29]},
    #     "max_len": 2800
    # }
    #
    # poses_list = amc_translator.poses_list()
    # process_seqs(poses_list)
    pose_seq = Sequence(0.008, 'pose')
    # pose_seq.data = poses_list[0]
    # pose_seq.re_sampling(0.032)
    # pose_seq.save(conf_path('tst_seqs/amc_13_29.json', 'tst'))
    pose_seq.load(conf_path('tst_seqs/amc_13_29.json', 'tst'))
    show_pose_seq_joints(pose_seq)


if __name__ == '__main__':
    get_pose_sequence()