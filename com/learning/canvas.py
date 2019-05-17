import tensorflow as tf
import os


class Canvas:

    file_name = 'model'
    path = ''
    sess = None

    def __init__(self, name=file_name, path=path):
        self.name = name
        self.path = path

    def open(self, path=None):
        tf.reset_default_graph()
        self.sess = tf.Session()
        if path is not None:
            self.load(path)
        return self.sess

    def save(self, path=None, flag=False, step=-1):
        if path is not None:
            self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        sess =self.sess
        saver = tf.train.Saver()
        # 如果图未保存，重新保存一次
        if not os.path.isfile(self.meta_file()):
            if step < 0:
                saver.save(sess, self.model_file())
            else:
                saver.save(sess, self.model_file(), global_step=step)
        # 如果flag为True，则必定重新保存图
        elif step < 0:
            saver.save(sess, self.model_file(), write_meta_graph=flag)
        # 如果step大于等于0，保存步数
        else:
            saver.save(sess, self.model_file(), global_step=step, write_meta_graph=flag)
        return self

    def save_all(self, path=None, step=-1):
        return self.save(path, True, step)

    def save_step(self, step, path=None, flag=False):
        return self.save(path, flag, step)

    def load_graph(self, path, graph):
        self.path = path
        g1 = tf.Graph()
        self.sess = tf.Session(graph=g1)
        with g1.as_default():
            saver = tf.train.import_meta_graph(self.meta_file())
            saver.restore(self.sess, self.model_file())
            graph.restore()

    def load(self, path=None, step=-1):
        if path is not None:
            self.path = path
        sess =self.sess
        saver = tf.train.import_meta_graph(self.meta_file())
        if step >= 0:
            saver.restore(sess, self.step_file(step))
        saver.restore(sess, self.model_file())
        return self

    def load_step(self, step, path=None):
        return self.load(path, step)

    def close(self):
        if self.sess is None:
            return
        self.sess.close()
        self.sess = None
        return self

    def __enter__(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        return self.sess

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def model_file(self):
        return os.path.join(self.path, self.file_name)

    def meta_file(self):
        return os.path.join(self.path, self.file_name + '.meta')

    def step_file(self, step):
        return os.path.join(self.path, self.file_name + '-%d' % step)
