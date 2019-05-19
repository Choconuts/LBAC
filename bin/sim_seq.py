import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from absl import app
from absl import flags
import os
from lbac.sequence.simulation import *
from com.path_helper import *


FLAGS = flags.FLAGS


flags.DEFINE_string('in', conf_path('seqs'), 'sequence out put dir, relative to db')
flags.DEFINE_string('sim', conf_path('sim'), 'sim dir')
flags.DEFINE_string('out', conf_path('extract'), 'data extracted, both meta and meshes')
flags.DEFINE_string('dir', 'rrr', 'r(relative) a(absolute), or d(relative to db), default d, in-out-ex')
flags.DEFINE_string('mesh', None, 'path to cloth mesh')
flags.DEFINE_string('type', 'pose', 'simulation type')
flags.DEFINE_integer('s', 0, 'start index')
flags.DEFINE_integer('e', -1, 'end index')
flags.DEFINE_integer('m', 1, 'mode, 0 for display, 1 for offline')
flags.DEFINE_integer('cloth', 0, 'cloth id in robe')
flags.DEFINE_bool('no_ext', False, 'only extract simulation results')


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

    out_dir = get_dir('out', 2)
    sim_dir = get_dir('sim', 1)
    in_dir = get_dir('in', 0)
    cloth_id = getattr(FLAGS, 'cloth')
    start = getattr(FLAGS, 's')
    end = getattr(FLAGS, 'e')
    mode = getattr(FLAGS, 'm')
    sim_type = getattr(FLAGS, 'type')

    if FLAGS.dir[0] == 'd':
        sim_dir = os.path.join(get_base(), sim_dir)
    seq_reader = SeqReader(in_dir)
    if end < 0:
        end = seq_reader.seq_num
    res = simulate(seq_reader, sim_dir, cloth_id, range(start, end), mode)
    if res and not getattr(FLAGS, 'no_ext'):
        extract_results(out_dir, sim_type)


if __name__ == '__main__':
    app.run(main)

