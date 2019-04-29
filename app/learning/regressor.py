from app.geometry.mesh import Mesh
import numpy as np
from app.learning.mlp import vertex_num, model_path, predict, shape_graph
from app.learning.gru import pose_model_path, predict_pose, pose_graph
from app.learning.ground_truths import beta_gt, pose_gt
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
        shape_graph()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        # saver.restore(self.sess, model_file)

    def feed(self, beta):
        self.displacement = np.reshape(predict(self.sess, beta), [vertex_num, 3])
        return self


class PoseRegressor(Regressor):
    def __init__(self, model_file=pose_model_path):
        self.model_path = model_file
        pose_graph()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)

    def feed(self, pose):
        self.displacement = np.reshape(predict_pose(self.sess, [pose])[0][0], [vertex_num, 3])
        return self


if __name__ == '__main__':
    pr = PoseRegressor()
    idx = 5
    beta = beta_gt.betas[0]
    pose = np.zeros((24, 3))
    pose = pose_gt.pose_seqs[0][10]
    disp = pose_gt.pose_disps[0][10]

    body = smpl.set_params(beta=beta)
    # print(0)
    pr.feed(pose).apply(beta_gt.template).save('../test/rebuild2.obj')
    # pr.displacement = disp
    cloth = pr.apply(beta_gt.template)
    smpl.set_params(pose=pose)
    from app.display.shape_display import apply_pose, VertexRelation
    cloth = apply_pose(cloth, body, pose, rela=VertexRelation().load("../display/shape/test_rela.json").get_rela())
    cloth.save('../test/rebuild3.obj')

    # pr.feed(pose).apply(beta_gt.template).save('../test/rebuild_pose1.obj')
    # print(1)
    # body.save('../test/rebuild2-body.obj')
    # rg.displacement = beta_gt.displacement[5]
    # cloth = rg.apply(beta_gt.template)
    # cloth.save('../test/beta_gt_cloth_5.obj')