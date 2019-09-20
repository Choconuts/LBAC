from com.learning.canvas import *
from com.learning.graph_helper import *
from lbac.train.ex_shape_gt import *
from lbac.display.virtual_fitting import *
from com.mesh.mesh import *
from com.mesh.smooth import *
from com.posture.smpl import *
from lbac.display.stage import *
import numpy as np


def gen_random_beta(r=2):
    beta = np.random.rand(4) * r * 2 - r
    print(beta)
    return beta


def batch(beta):
    return [[beta], 1, None]


def init():
    smpl = SMPLModel(conf_path('model/smpl/male.pkl'))
    canvas = Canvas()
    sess = canvas.open('../../db/model/ex/beta/6')
    g = GraphBase('mlp').restore()
    cloth = Mesh().load('../../db/gt/ex/beta/2019-6-19/template.obj')
    beta = gen_random_beta(2)
    diff = g.predict(sess, batch(beta)).reshape(-1, 3)

    # beta_gt = BetaGroundTruth().load('../../db/gt/ex/beta/2019-6-19')
    # diff = np.array(beta_gt.data['disps']['0']).reshape((-1, 3))

    cloth.vertices += diff
    cloth.update()
    beta0 = np.zeros(10)
    beta0[0:4] = beta
    pose0 = np.zeros((24, 3))
    pose0[0] = [1.579, 0, 0]
    trans0 = np.zeros(3)
    trans0 += [0, 0.2, -0.357]
    print(beta0)
    smpl.set_params(beta=beta0, pose=pose0, trans=trans0)
    body = Mesh()
    body.vertices = smpl.verts
    body.faces = smpl.faces
    body.save(conf_path('body.obj', 'tst'))
    cloth.save(conf_path('rebuild.obj', 'tst'))
    canvas.close()


def save_verts(array, txt):
    lines = []
    for v in array:
        line = ''
        for x in v:
            line += str(x) + ' '
        line = line[:-1] + '\n'
        lines.append(line)
    with open(txt, 'w') as fp:
        fp.writelines(lines)


base_dir = 'rands2'
conf_json = 'conf.json'
conf = {}
if not exists(conf_json):
    save_json(conf, conf_json)
conf = load_json(conf_json)
if base_dir not in conf:
    conf[base_dir] = {}


def get_id():
    if 'next_id' not in conf[base_dir]:
        conf[base_dir]['next_id'] = 0
    conf[base_dir]['next_id'] += 1
    return conf[base_dir]['next_id'] - 1


def gen_random():

    smpl = SMPLModel(conf_path('model/smpl/male.pkl'))
    canvas = Canvas()
    sess = canvas.open('../../db/model/ex/beta/6')
    g = GraphBase('mlp').restore()
    cloth = Mesh().load('../../db/gt/ex/beta/2019-6-19/template.obj')
    beta = gen_random_beta(2)
    diff = g.predict(sess, batch(beta)).reshape(-1, 3)

    cloth.vertices += diff
    cloth.update()
    beta0 = np.zeros(10)
    beta0[0:4] = beta
    pose0 = np.zeros((24, 3))
    pose0[0] = [1.579, 0, 0]
    trans0 = np.zeros(3)
    trans0 += [0, 0.2, -0.357]
    print(beta0)
    smpl.set_params(beta=beta0, pose=pose0, trans=trans0)
    body = Mesh().from_vertices(smpl.verts, smpl.faces)
    canvas.close()

    return beta, cloth, body


def get_relation(cloth=None, body=None):
    if 'default_rela' not in conf:
        conf['default_rela'] = 'rela.json'
    rela_path = conf['default_rela']
    if not exists(rela_path):
        body.update()
        rela = ClosestVertex().calc(cloth, body).save(rela_path)
    else:
        rela = ClosestVertex().load(rela_path)
        return rela.calc_rela_once(cloth, body)

    return rela.get_rela()


def post_processing(body, cloth):
    rela = get_relation(cloth, body)

    def spread(ci, vec, levels=40):
        past = {}
        bias = 0.002

        # def get_vec(i, level):
        #     return vec * bias / (np.linalg.norm(cloth.vertices[i] - cloth.vertices[ci]) + bias)
        #     # return vec * level / tot_level

        def forward(i, level):
            if level == 0 or i in past:
                return
            past[i] = 1
            cloth.vertices[i] += vec * bias / (np.linalg.norm(cloth.vertices[i] - cloth.vertices[ci]) + bias)
            for ni in cloth.edges[i]:
                forward(ni, level - 1)

        forward(ci, levels)

    # to the close skin
    for i in range(len(cloth.vertices)):
        vc = cloth.vertices[i]
        vb = body.vertices[rela[i]]
        bn = body.normal[rela[i]]
        if np.dot(vb - vc, bn) > 0:
            spread(i, bn * np.dot(vb - vc, bn) * 0.5, 20)

    for r in range(2000):
        flag = True
        for i in range(len(cloth.vertices)):
            vc = cloth.vertices[i]
            vb = body.vertices[rela[i]]
            bn = body.normal[rela[i]]
            if np.dot(vb - vc, bn) > 0.004:
                spread(i, bn * np.dot(vb - vc, bn) * 1, 10)
                flag = False
        if flag:
            break

    cloth.update_normal_only()
    for i in range(len(cloth.vertices)):
        cn = cloth.normal[i]
        bn = body.normal[rela[i]]
        cloth.vertices[i] += cn * 0.004 + bn * 0.004


def save_sample(beta, cloth, body):
    id = str3(get_id())
    if exists(join(base_dir, id)):
        os.rmdir(join(base_dir, id))
    os.makedirs(join(base_dir, id))
    save_verts(cloth.vertices, join(base_dir, id, 'cloth.txt'))
    save_verts(body.vertices, join(base_dir, id, 'body.txt'))
    save_verts(np.reshape(beta, (-1, 1)), join(base_dir, id, 'beta.txt'))

    odir = join(base_dir, 'obj-' + id)
    if exists(odir):
        os.rmdir(odir)
    os.makedirs(odir)
    cloth.save(join(odir, 'cloth.obj'))
    body.save(join(odir, 'body.obj'))


if __name__ == '__main__':
    # init()
    random.seed(2313)
    for i in range(1):
        beta, cloth, body = gen_random()
        body.vertices += [0., 0.03, 0.005]
        # post_processing(body, cloth)
        view_meshes((body, cloth))
        save_sample(beta, cloth, body)


save_json(conf, conf_json)

