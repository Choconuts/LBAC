from absl import app
from absl import flags
import os


FLAGS = flags.FLAGS

flags.DEFINE_string('gpu', None, 'comma separated list of GPU(s) to use.')
flags.DEFINE_bool('x', False, '123')


def main(argv):
    del argv
    print(FLAGS.x)
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    else:
        print('Please assign GPUs.')
        exit()


if __name__ == '__main__':
    app.run(main)

