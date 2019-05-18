from absl import app
from absl import flags
import os
from com.path_helper import *
from lbac.train.shape_gt import gen_beta_gt_data, set_smooth_times

FLAGS = flags.FLAGS

flags.DEFINE_string('in', conf_path('extract'), 'simulation out put dir, relative to db')
flags.DEFINE_string('out', conf_path('gt'), 'ground truth dir')
flags.DEFINE_string('dir', 'dd', 'r(relative) a(absolute), or d(relative to db), default d, in-out-ex')

flags.DEFINE_integer('smooth', -1, 'gen default')

flags.DEFINE_string('m', 's', 'gen default')
flags.DEFINE_bool('s', False, 'gen shape')
flags.DEFINE_bool('p', False, 'gen shape')


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

    shape_flag = getattr(FLAGS, 's')
    pose_flag = getattr(FLAGS, 'p')
    mode = getattr(FLAGS, 'm')
    smooth_factor = getattr(FLAGS, 'smooth')
    if smooth_factor >= 0:
        set_smooth_times(smooth_factor)

    if shape_flag:
        gen_beta_gt_data(in_dir, out_dir)


if __name__ == '__main__':
    app.run(main)


