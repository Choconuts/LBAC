from absl import app
from absl import flags
import os
from com.path_helper import *

FLAGS = flags.FLAGS

flags.DEFINE_string('in', conf_path('sim'), 'simulation out put dir, relative to db')
flags.DEFINE_string('seq', conf_path('seqs'), 'sequence out put dir, relative to db')
flags.DEFINE_string('ext', conf_path('extract'), 'extracted data dir')
flags.DEFINE_string('out', conf_path('gt'), 'ground truth dir')
flags.DEFINE_string('dir', 'dd', 'r(relative) a(absolute), or d(relative to db), default d, in-out-ex')
flags.DEFINE_string('mesh', None, 'path to cloth mesh')
flags.DEFINE_integer('s', 0, 'start index')
flags.DEFINE_integer('e', -1, 'end index')
flags.DEFINE_integer('m', 1, 'end index')
flags.DEFINE_integer('cloth', 0, 'cloth id in robe')


def get_dir(key, i):
    m_dir = getattr(FLAGS, key)
    if not FLAGS.dir or len(FLAGS.dir) < 1:
        return m_dir
    if i >= len(FLAGS.dir):
        i = len(FLAGS.dir) - 1
    if FLAGS.dir[i] == 'd':
        m_dir = os.path.join(get_base('db'), m_dir)
    return m_dir


def main(argv):
    del argv

    out_dir = get_dir('out', 1)
    in_dir = get_dir('in', 0)
    cloth_id = getattr(FLAGS, 'cloth')
    start = getattr(FLAGS, 's')
    end = getattr(FLAGS, 'e')
    mode = getattr(FLAGS, 'm')

    if FLAGS.dir[0] == 'd':
        out_dir = os.path.join(get_base('db'), out_dir)
    seq_reader = SeqReader(in_dir)
    if end < 0:
        end = seq_reader.seq_num
    simulate(seq_reader, out_dir, cloth_id, range(start, end), mode)


if __name__ == '__main__':
    app.run(main)


