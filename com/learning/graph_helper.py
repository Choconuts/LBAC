import tensorflow as tf
import os
from com.learning.canvas import Canvas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# TODO


class GraphBase:

    id = '$'

    trainers = []
    inputs = []
    parameters = []
    evaluates = []
    outputs = []
    interfaces = ['trainers', 'inputs', 'parameters', 'evaluates', 'outputs']

    iter = 1000
    show_steps = 50
    batch_size = 128
    save_step = 100

    def __init__(self, name):
        self.id = str(name)
        self.name_pool = dict()
        for collection in self.interfaces:
            setattr(self, collection, [])
        self.global_step = 0

    # 外部方法

    def generate(self):
        g = self.graph_func()
        for i in range(len(g)):
            setattr(self, self.interfaces[i], self.collect(g[i], self.interfaces[i]))
        return self

    def restore(self, index=0, global_step=0):
        for i in range(len(self.interfaces)):
            j = 0
            res = []
            while True:
                tensors = tf.get_collection(self.get_name(self.interfaces[i], j))
                if len(tensors) == 0:
                    break
                else:
                    res.append(tensors[index])
                j += 1
            setattr(self, self.interfaces[i], res)
        self.global_step = global_step
        return self

    def set_steps(self, iteration, display=show_steps, saving=save_step):
        self.iter = iteration
        self.show_steps = display
        self.save_step = saving
        return self

    def set_callback(self, b_process=None, p_process=None, saving=None):
        if b_process is not None:
            self.batch_process = b_process
        if p_process is not None:
            self.predict_process = p_process
        if saving is not None:
            self.save_graph = saving

    # 需要重写/外部修改的方法

    def graph_func(self):
        return sample_graph()

    def batch_process(self, raw_batch, is_train, local_step):
        return raw_batch

    def predict_process(self, raw_batch, is_test):
        return raw_batch

    def save_graph(self):
        pass

    def train(self, canvas: Canvas, ground_truth, iteration=-1, test=False, initialize=False):
        sess = canvas.sess
        if iteration < 0:
            iteration = self.iter
        if initialize:
            sess.run(tf.global_variables_initializer())

        for i in range(iteration):
            self.global_step += 1
            self.train_step(sess, ground_truth.get_batch(self.batch_size), i + 1)
            if (i + 1) % self.show_steps == 0:
                self.eval_step(sess, ground_truth.get_batch(self.batch_size), i + 1)
            if self.save_step is not None and self.save_step > 0 and (i + 1) % self.save_step == 0:
                self.save_graph()
                canvas.save_step(self.global_step)

        if test:
            self.test_step(sess, ground_truth.get_test())

    def predict(self, sess: tf.Session, batch, index=0):
        feed_dict = self.pack(self.predict_process(batch, False))
        return sess.run(self.outputs[index], feed_dict=feed_dict)

    # 可以内部利用的方法

    def train_step(self, sess, batch, step):
        feed_dict = self.pack(self.batch_process(batch, True, step))
        if feed_dict is None:
            print('train failes')
            return
        for t in self.trainers:
            sess.run(t, feed_dict=feed_dict)

    def eval_step(self, sess, batch, step):
        feed_dict = self.pack(self.batch_process(batch, False, step))
        if feed_dict is None:
            print('eval failed')
            return
        for e in self.evaluates:
            ev = sess.run(e, feed_dict=feed_dict)
            print('step {}, evaluation: {}'.format(step, ev))

    def test_step(self, sess, batch, step=0):
        feed_dict = self.pack(self.predict_process(batch, True))
        if feed_dict is None:
            print('tst failed')
            return
        for e in self.evaluates:
            ev = sess.run(e, feed_dict=feed_dict)
            print('test step {}, evaluation: {}'.format(step, ev))

    # 用不到的方法

    def build_name(self, collection):
        if collection not in self.name_pool:
            self.name_pool[collection] = 0
        else:
            self.name_pool[collection] += 1
        return self.get_name(collection, self.name_pool[collection])

    def get_name(self, collection, i):
        return str(self.id + '_' + str(collection) + '_' + str(i))

    def collect(self, tensors, collection):
        if tensors is None:
            return []
        if type(tensors) != list and type(tensors) != tuple:
            tf.add_to_collection(self.build_name(collection), tensors)
            return [tensors]
        for tensor in tensors:
            tf.add_to_collection(self.build_name(collection), tensor)
        return tensors

    def pack(self, batch):
        if batch is None or len(batch) == 0:
            return None
        feed = dict()

        for i in range(len(batch)):
            if batch[i] is None:
                continue
            feed[self.inputs[i]] = batch[i]
        return feed


def sample_graph():
    x = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='x')
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='y')

    o = tf.contrib.layers.fully_connected(inputs=x, num_outputs=1, activation_fn=None)
    c = tf.reduce_mean(tf.square(o - y))
    t = tf.train.AdamOptimizer(1e-2).minimize(c)

    return t, [x, y], None, c, o
