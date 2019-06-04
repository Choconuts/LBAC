import numpy as np
import pickle


class SMPLModel():
    def __init__(self, model_path):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

            self.J_regressor = params['J_regressor']
            self.weights = params['weights']
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']
            self.G = np.array([])

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None, no_return=False, mat_only=False):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose is not None:
          self.pose = pose
        if beta is not None:
          self.beta = beta
        if trans is not None:
          self.trans = trans
        if mat_only:
            self.update_R_and_G()
        else:
            self.update()
        if no_return:
            return

    def update_R_and_G(self):
        # rotation matrix for each joint
        self.R = self.rodrigues(self.pose.reshape((-1, 1, 3)))

        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        self.G = G

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        self.v_shaped = v_shaped
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))

        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)

        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0]-1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)

        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
                    )
                  )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
                )
            )
        self.G = G
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        # T (6890, 4, 4)
        # use stack to make homogeneous coordinates
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        # use slice to get original coordinates
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        self.verts = v + self.trans.reshape([1, 3])

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta

        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def apply(smpl, weights, v_posed):
    T = np.tensordot(weights, smpl.G, axes=[[1], [0]]) # 6000 * 23 x 23 * 4 * 4 = 6000 * 4 * 4
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))

    return np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3] + smpl.trans.reshape([1, 3])


def dis_apply(smpl, weights, vertices):
    T = np.tensordot(weights, smpl.G, axes=[[1], [0]])
    v = vertices - smpl.trans.reshape([1, 3]) # 7300 * 3
    v_h = np.hstack((v, np.ones([v.shape[0], 1]))).reshape([-1, 4, 1])  # 7300 * 4 * 1
    T_i =np.linalg.inv(T) # 7300 * 4 * 4
    v_posed = np.matmul(T_i, v_h).reshape([-1, 4])
    v_posed = v_posed[:, :3]
    return v_posed


if __name__ == '__main__':
    np.random.seed(1212)
    pose = []
    for i in range(24):
        pose.append([0.1, 0.1, 0.1])
    pose = np.array(pose)
    print(type(pose))
    beta = []
    for i in range(10):
        beta.append(0)
    beta[3] = 2
    beta = np.array(beta)
    print(beta)


