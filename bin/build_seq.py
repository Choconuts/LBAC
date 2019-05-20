import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from absl import app
from absl import flags
from lbac.sequence.builder import *
from lbac.sequence.pose_translator import JsonTranslator, AmcTranslator
from com.path_helper import *

FLAGS = flags.FLAGS

flags.DEFINE_string('out', 'seq\cmu_1', 'sequence out put dir, relative to db')   # conf_path('seqs')
flags.DEFINE_string('dir', 'rd', 'r(relative) a(absolute), or d(relative to db), default d, in-out-ex')
flags.DEFINE_string('in', conf_path('seq_poses_json'), 'input poses json file, list of sequences')
flags.DEFINE_integer('lerp', 10, 'interpolate frames num')
flags.DEFINE_integer('shape', 1, 'the first one, or first 17 shapes')
flags.DEFINE_string('type', 'a', 'shape or pose, or pose from amc')         # s


def get_dir(key, i):
    m_dir = getattr(FLAGS, key)
    if not FLAGS.dir or len(FLAGS.dir) < 1:
        return m_dir
    if i >= len(FLAGS.dir):
        i = len(FLAGS.dir) - 1
    if FLAGS.dir[i] == 'd':
        m_dir = os.path.join(get_base(), m_dir)
    return m_dir


def main(argv):
    del argv

    out_dir = get_dir('out', 1)

    if FLAGS.type == 's':
        set_smpl(SMPLModel(conf_path('smpl')))
        build_17_betas_sequence(out_dir, FLAGS.lerp)
    elif FLAGS.type in ['p', 'a', 'j']:
        if FLAGS.type == 'a':
            set_default_translator(AmcTranslator())
        if FLAGS.type == 'j':
            set_default_translator(JsonTranslator())
        in_dir = get_dir('in', 0)
        set_smpl(SMPLModel(conf_path('smpl')))
        build_poses_sequence(out_dir, in_dir, range(FLAGS.shape), FLAGS.lerp)


if __name__ == '__main__':
    app.run(main)

