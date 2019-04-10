from app.geometry.mesh import Mesh
import numpy as np
from app.learning.mlp import vertex_num, model_path, predict
from app.geometry.smooth import smooth
import tensorflow as tf

from app.smpl.smpl_np import SMPLModel, smpl


class Regressor:
    displacement = np.array([])

    def apply(self, mesh):
        mesh.vertices += self.displacement
        return mesh


class ShapeRegressor(Regressor):

    def __init__(self, model_file=model_path):
        self.model_path = model_file
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)

    def feed(self, beta):
        self.displacement = np.reshape(predict(self.sess, beta), [vertex_num, 3])
        return self


if __name__ == '__main__':
    rg = ShapeRegressor()
    beta = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    body = smpl.set_params(beta=beta)
    print(0)
    Mesh().load('../data/beta_simulation/avg_smooth.obj').save('../test/rebuild1.obj')
    smooth(rg.feed(beta).apply(Mesh().load('../data/beta_simulation/avg_smooth.obj')), 2).save('../test/rebuild1.obj')
    print(1)
    body.save('../test/rebuild1-body.obj')