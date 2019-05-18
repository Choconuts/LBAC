from absl import app
from absl import flags
import os
from lbac.sequence.builder import *
from com.path_helper import *


FLAGS = flags.FLAGS


flags.DEFINE_string('in', conf_path('seqs'), 'sequence out put dir, relative to db')
flags.DEFINE_string('out', conf_path('sim'), 'sim dir')
flags.DEFINE_string('dir', 'dd', 'r(relative) a(absolute), or d(relative to db), default d, in-out-ex')
flags.DEFINE_integer('lerp', 20, 'interpolate frames num')
flags.DEFINE_integer('shape', 1, 'the first one, or first 17 shapes')
flags.DEFINE_string('type', 's', 'shape or pose, or other')


def get_dir(key, i):
    m_dir = getattr(FLAGS, key)
    out_dir = FLAGS.out
    if not FLAGS.dir or len(FLAGS.dir) < 1:
        return out_dir
    if i >= len(FLAGS.dir):
        i = len(FLAGS.dir) - 1
    if FLAGS.dir[i] == 'd':
        m_dir = os.path.join(get_base('db'), out_dir)
    return m_dir


def main(argv):
    del argv

    out_dir = get_dir('out', 0)

    in_dir = get_dir('in', 1)

    if FLAGS.dir[0] == 'd':
        out_dir = os.path.join(get_base('db'), out_dir)

    if FLAGS.type == 's':
        set_smpl(SMPLModel(conf_path('smpl')))
        build_17_betas_sequence(FLAGS.out)
    elif FLAGS.type == 'p':
        set_smpl(SMPLModel(conf_path('smpl')))
        build_poses_sequence(out_dir, in_dir, range(FLAGS.shape), FLAGS.lerp)


if __name__ == '__main__':
    app.run(main)

