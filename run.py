from absl import app
from absl import flags

import sys, os

FLAGS = flags.FLAGS

flags.DEFINE_string('run', 'seq', 'python to run')   # conf_path('seqs')

self = sys.modules[__name__]


def run(argv):

    pgm = FLAGS.run
    if pgm == 'seq':
        from bin.build_seq import main
    elif pgm == 'gt':
        from bin.gen_gt import main
    elif pgm == 'sim':
        from bin.sim_seq import main
    else:
        def main(argv):
            del argv
            print('please enter program name')

    main(argv)

if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))

    app.run(run)