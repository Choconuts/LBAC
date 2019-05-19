from com.learning import mlp
from absl import app
from absl import flags
import os
from com.path_helper import *
from lbac.train.shape_gt import gen_beta_gt_data, set_smooth_times
from lbac.train.pose_gt import gen_pose_gt_data
from com.learning.canvas import Canvas

FLAGS = flags.FLAGS
flags.DEFINE_string('out', conf_path('mlp'), 'out put dir')
flags.DEFINE_string('out', conf_path('mlp'), 'out put dir')

flags.DEFINE_integer('input', 4, 'start index')
flags.DEFINE_integer('output', 7366 * 3, 'start index')
flags.DEFINE_multi_integer('hidden', [20], 'start index')
flags.DEFINE_integer('batch', 128, 'start index')
flags.DEFINE_integer('show', 100, 'start index')
flags.DEFINE_integer('decay', 500, 'start index')
flags.DEFINE_float('rate', 1e-3, 'start index')
flags.DEFINE_float('keep', 0.8, 'start index')
flags.DEFINE_bool('test', False, 'start index')


def train(canvas=None):
    mlp.n_input = FLAGS.input
    mlp.n_output = FLAGS.output
    mlp.hidden = FLAGS.hidden
    mlp.batch_size = FLAGS.batch
    mlp.learning_rate = FLAGS.rate
    mlp.decay_step = FLAGS.decay
    mlp.keep_probability = FLAGS.keep
    mlp.show_step = FLAGS.show

    if canvas is None:
        canvas = Canvas()
    canvas.open()