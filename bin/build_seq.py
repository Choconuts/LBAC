from absl import app
from absl import flags
import os
from lbac.sequence.builder import *


FLAGS = flags.FLAGS

flags.DEFINE_string('out', None, 'sequence out put dir, relative to db')
flags.DEFINE_string('dir', None, 'r(relative) a(absolute), or d(relative to db), default d')
flags.DEFINE_string('in', None, 'input poses json file, list of sequences')
flags.DEFINE_integer('lerp', 20, 'interpolate frames num')
flags.DEFINE_integer('shape', 1, 'the first one, or first 17 shapes')
flags.DEFINE_string('type', 's', 'shape or pose, or other')


def main(argv):
    del argv
    if FLAGS.type == 's':
        set_smpl(SMPLModel(conf_path('smpl')))
        build_17_betas_sequence(FLAGS.out)
    elif FLAGS.type == 'p':
        set_smpl(SMPLModel(conf_path('smpl')))
        build_poses_sequence(FLAGS.out, getattr(FLAGS, 'in'), range(FLAGS.shape), FLAGS.lerp)


if __name__ == '__main__':
    app.run(main)

