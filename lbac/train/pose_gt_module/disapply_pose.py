from com.learning.ground_truth import *
from com.path_helper import *
from com.mesh.smooth import *
from lbac.train.shape_gt import BetaGroundTruth, parse_sim, parse_ext
from com.mesh.closest_vertex import ClosestVertex, VertexMapping
from com.posture.smpl import *

relation = ClosestVertex()
weights = None


def get_cloth_weights(cloth: Mesh, body: Mesh, pose, smpl: SMPLModel, beta):
    relation.load(conf_path('vert_rela'))
    smpl.set_params(pose=np.array(pose))

    # relation.calc_rela_once(cloth, body)
    rela = relation.get_rela()
    body_weights = smpl.weights
    cloth_weights = np.zeros((len(cloth.vertices), 24))

    for i in range(len(cloth.vertices)):
        cloth_weights[i] = body_weights[rela[i]]
    return cloth_weights


def set_getter(func):
    global get_cloth_weights
    get_cloth_weights = func


def load_weights():
    global weights
    if weights is None:
        weights = np.array(load_json(conf_path(r'model\relation\cloth_weights_7366.json')))

    return weights


set_getter(load_weights)