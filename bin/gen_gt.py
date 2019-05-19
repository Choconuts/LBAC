import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from absl import app
from absl import flags
import os
from com.path_helper import *
from lbac.train.shape_gt import gen_beta_gt_data, set_smooth_times
from lbac.train.pose_gt import gen_pose_gt_data

FLAGS = flags.FLAGS

flags.DEFINE_string('in', conf_path('extract'), 'simulation out put dir, relative to db')
flags.DEFINE_string('beta', conf_path('b_gt'), 'ground truth dir')
flags.DEFINE_string('pose', conf_path('p_gt'), 'ground truth dir')
flags.DEFINE_string('dir', 'rrr', 'r(relative) a(absolute), or d(relative to db), default d, in-out-ex')

flags.DEFINE_integer('smooth', -1, 'gen default')
flags.DEFINE_integer('s', 0, 'start index')
flags.DEFINE_integer('e', -1, 'end index')

flags.DEFINE_string('m', 'b', 'gen default')
flags.DEFINE_bool('b', False, 'gen shape')
flags.DEFINE_bool('p', False, 'gen shape')


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

    beta_gt_dir = get_dir('beta', 1)
    pose_gt_dir = get_dir('pose', 1)
    in_dir = get_dir('in', 0)

    shape_flag = getattr(FLAGS, 'b')
    pose_flag = getattr(FLAGS, 'p')
    start = getattr(FLAGS, 's')
    end = getattr(FLAGS, 'e')
    if end > start:
        gen_range = range(start, end)
    else:
        gen_range = None

    mode = getattr(FLAGS, 'm')
    smooth_factor = getattr(FLAGS, 'smooth')
    if smooth_factor >= 0:
        set_smooth_times(smooth_factor)

    if not shape_flag and not pose_flag:
        shape_flag = 'b' in mode
        pose_flag = 'p' in mode

    if shape_flag:
        gen_beta_gt_data(in_dir, beta_gt_dir)

    if pose_flag:
        gen_pose_gt_data(in_dir, beta_gt_dir, pose_gt_dir, gen_range)


if __name__ == '__main__':
    app.run(main)


