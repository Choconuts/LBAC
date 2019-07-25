# -*- coding:utf-8 -*-
from com.sequence.sequence import *
from lbac.train.pose_gt import *
from lbac.display.seq_show import *



def test_seq():
    seq = Sequence(0.1, 'test')
    seq.data = np.linspace([1, 4], [10, 13], 10)
    seq.save('../../tst/tstseq.json')
    seq.load('../../tst/tstseq.json')
    # print(seq.data)
    assert seq.get_frame_num() == 10
    assert seq.frame_name(1) == '00001'
    assert seq.time_step == 0.1
    assert seq.type == 'test'
    assert seq.get_frame_rate() == 10
    assert seq.get_total_time() == 1
    # print(seq.get_shot_at(0.89))
    seq2 = seq.copy()
    seq.data[0] *= 0
    assert seq2.data[0][1] != 0
    seq2.re_sampling(0.05)
    seq2.data = seq2.data[1:]
    seq2.re_sampling(0.1)
    print(seq2.data)
    seq.slice(0.05, 0.7)
    print(seq.data)


def test_show_seq():
    path = '../../tst/test_show_pose_seq2.json'
    if not exists(path):
        pose_gt = PoseGroundTruth().load(conf_path('gt/pose/1'))
        obj = pose_gt.data[list(pose_gt.index.keys())[1]]
        pose_seq = Sequence(0.033, 'pose')
        pose_seq.meta['beta'] = obj['beta']
        pose_seq.data = obj['poses']
        pose_seq.save(path)
    pose_seq = Sequence(0.033, 'pose').load(path)
    # show_pose_seq(pose_seq)




if __name__ == '__main__':
    test_seq()