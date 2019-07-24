# -*- coding:utf-8 -*-
from com.sequence.sequence import *


def test_seq():
    seq = Sequence(0.1, 'test')
    seq.data = np.linspace([1, 4], [10, 13], 10)
    seq.save('../../tst/tstseq.json')
    seq.load('../../tst/tstseq.json')
    print(seq.data)
    assert seq.get_frame_num() == 10
    assert seq.frame_name(1) == '00001'
    assert seq.time_step == 0.1
    assert seq.type == 'test'
    assert seq.get_frame_rate() == 10
    assert seq.get_total_time() == 1
    print(seq.get_shot_at(0.89))


if __name__ == '__main__':
    test_seq()