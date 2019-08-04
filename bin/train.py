import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from absl import app
from absl import flags
import os
from com.path_helper import *
from com.learning.canvas import Canvas
from com.timer import timing

FLAGS = flags.FLAGS

flags.DEFINE_string('conf', join(get_base('conf'), 'train.json'), 'default configuration')
flags.DEFINE_string('c', None, 'edit config')
flags.DEFINE_bool('t', False, 'train or not')
flags.DEFINE_integer('l', None, 'loading and continue')
flags.DEFINE_string('out', None, 'out put dir')
flags.DEFINE_string('dir', 'rrr', 'r(relative) a(absolute), or d(relative to db), default d, in-out-conf')

flags.DEFINE_string('gt_dir',None, 'input gt dir')
flags.DEFINE_string('gt',None, 'input gt type')
flags.DEFINE_integer('input', None, 'start index')
flags.DEFINE_integer('output', None, 'start index')
flags.DEFINE_integer('batch', None, 'start index')
flags.DEFINE_integer('iter', None, 'iter times')
flags.DEFINE_integer('show', None, 'start index')
flags.DEFINE_integer('decay', None, 'start index')
flags.DEFINE_float('rate', None, 'start index')
flags.DEFINE_float('keep', None, 'start index')
flags.DEFINE_bool('test', None, 'start index')
flags.DEFINE_integer('test_cut', None, 'start index')
flags.DEFINE_bool('no_test', None, 'start index')
flags.DEFINE_string('name', None, 'start index')
flags.DEFINE_float('l1', 0, 'regular')
flags.DEFINE_float('l2', 0, 'regular')
flags.DEFINE_multi_integer('hidden', None, 'start index')
flags.DEFINE_integer('rnn_step', None, 'rnn step')
flags.DEFINE_integer('save', None, 'save step')

flags.DEFINE_multi_string('g', ['gru'], 'which to load')

mapping = {
    "n_input": "input",
    "n_output": "output",
    "n_hidden": "hidden",
    "iter": "iter",
    "batch_size": "batch",
    "learning_rate": "rate",
    "decay_step": "decay",
    "n_steps": "rnn_step",
    "save_step": "save",
    "keep_probability": "keep",
    "show_step": "show",
    "test_flag": "test",
    "graph_id": "name",
    "gt": "gt",
    "l1_regular_scale": "l1",
    "l2_regular_scale": "l2",
    "test_cut": "test_cut"
}

def get_dir(key, i):
    m_dir = getattr(FLAGS, key)
    if not FLAGS.dir or len(FLAGS.dir) < 1:
        return m_dir
    if i >= len(FLAGS.dir):
        i = len(FLAGS.dir) - 1
    if FLAGS.dir[i] == 'd':
        m_dir = os.path.join(get_base(), m_dir)
    return m_dir


@timing
def main(argv):
    del argv

    edit_config_file = FLAGS.c
    train_flag = FLAGS.t
    test_cut = FLAGS.test_cut
    if edit_config_file and exists(edit_config_file):
        config = load_json(conf_path(edit_config_file + '.json','conf'))
    else:
        config = load_json(get_dir('conf', 2))
    graphs = FLAGS.g
    output_dir = get_dir('out', 1)
    load_step = FLAGS.l

    if config is None:
        config = dict()
    for graph in graphs:
        if graph not in config:
            config[graph] = dict()
        values = mapping.values()
        for v in values:
            change = getattr(FLAGS, v)
            if change is None and v == 'test':
                config[graph][v] = False
            if change is not None:
                if v == 'gt':
                    if not hasattr(FLAGS, 'gt_dir'):
                        print('warning: no gt dir entered')
                    change = [change, get_dir('gt_dir', 0)]
                config[graph][v] = change

        if getattr(FLAGS, 'no_test'):
            config[graph]['test'] = False

        print('config: ', config[graph])

    if edit_config_file:
        save_json(config, conf_path(edit_config_file + '.json','conf'))
    if train_flag:
        def set_attributes(module):
            for key in mapping:
                if mapping[key] in config[graph]:
                    setattr(module, key, config[graph][mapping[key]])

        def train(module, gt):
            set_attributes(module)
            canvas = Canvas()
            canvas.open()
            if load_step is not None:
                canvas.load_step(load_step, output_dir)

            g = module.Graph()
            if load_step is not None:
                g.restore()
                g.global_step = load_step
            else:
                g.generate()
            canvas.path = output_dir

            g.train(canvas, gt, test=module.test_flag, initialize=load_step is None)
            if output_dir is not None:
                canvas.save_all()

        import lbac.train.train_helper as th

        for graph in graphs:
            module = getattr(th, graph)
            print('config: ', config[graph])
            gt_type = getattr(th, str(config[graph]['gt'][0]))
            if "rnn_step" in config[graph]:
                if "test_cut" in config[graph]:
                    gt = gt_type(step=config[graph]['rnn_step'], cut_off=test_cut)
                else:
                    gt = gt_type(step=config[graph]['rnn_step'])
            elif "test_cut" in config[graph]:
                gt = gt_type(cut=test_cut)
            else:
                gt = gt_type()
            train(module, gt.load(config[graph]['gt'][1]))


if __name__ == '__main__':
    app.run(main)





