from com.learning.graph_helper import *
from com.path_helper import *
from lbac.train.auto_encoder_mesh_gt import txt_to_array_3, txt_to_array
from com.mesh.mesh import *
from com.learning import auto_enc
from com.learning.ground_truth import *
from com.timer import Timer


tst_model = r'/Users/choconut/Desktop/edu/CAD/project/LBAC/db/model/auto_enc/5'
ex_dir = r'/Users/choconut/Desktop/edu/CAD/project/LBAC/db/raw/pants_mesh/'
cloth_dir = 'shapeModel_lp0 1'
topo_file = 'triangles_clt.txt'

faces = []


def train():
    canvas = Canvas()
    canvas.open()
    auto_enc.n_input = 10
    auto_enc.n_hidden = [5, 3]
    auto_enc.learning_rate = 0.1
    enc = auto_enc.Graph().generate()

    class TstGT(GroundTruth):
        def get_batch(self, size):
            res = []
            for i in range(size):
                res.append([])
                for j in range(10):
                    res[i].append(
                        j * 0.07 + random.random() * 0.01
                    )

            return [np.array(res)]

    enc.save_step = -1

    enc.train(canvas, TstGT(), 10000, initialize=True)

    canvas.close()


class AutoEncoderTester:

    def assign(self):
        g = self.graph
        c = self.canvas
        for i in range(3, 7):
            c.sess.run(g.parameters[i], feed_dict={
                g.parameters[2]: np.zeros((1, self.code_size)),
                g.inputs[1]: 1
            })

    def __init__(self, graph: auto_enc.Graph, canvas: Canvas, code_size):
        self.graph = graph
        self.canvas = canvas
        self.code_size = code_size
        self.assign()

    def encode(self, x):
        g = self.graph
        c = self.canvas
        return c.sess.run(g.parameters[0], feed_dict={
            g.inputs[0]: np.array(x).reshape(-1, np.size(x)),
            g.inputs[1]: 1
        })

    def decode(self, code):
        g = self.graph
        c = self.canvas
        return c.sess.run(g.parameters[7], feed_dict={
            g.parameters[0]: np.array(code).reshape(-1, self.code_size),
            g.inputs[1]: 1
        })


def tst():

    def get_verts(i):
        cloth_file = 'cloth_' + str3(i) + '.txt'
        cloth_file = join(ex_dir, cloth_dir, cloth_file)
        vertices = txt_to_array_3(cloth_file, 'f')
        return vertices

    def get_faces():
        global faces
        if faces is None or len(faces) == 0:
            faces = txt_to_array_3(join(ex_dir, topo_file), 'i')
            # faces -= 1
        return faces

    def get_mesh(i):
        mesh = Mesh().from_vertices(get_verts(i), get_faces())
        mesh.update()
        return mesh

    vts = get_verts(3)
    g = auto_enc.Graph()
    vts_arr = vts.reshape([-1, 35064])
    canvas = Canvas()
    canvas.load_graph_of_step(tst_model, g, 5000)

    aet = AutoEncoderTester(g, canvas, 5)
    ecded = aet.encode(vts_arr)
    print(ecded)
    ecded = ecded * 0.9
    tm = Timer()
    dcded = aet.decode(ecded.repeat(100, 0))
    tm.tick('decode')
    dcded = dcded.reshape(-1, 3)
    res = g.predict(canvas.sess, [vts_arr])
    print(np.mean(np.square(res - vts_arr)))

    new_vts = np.reshape(np.array(res), (len(vts), 3))
    Mesh().from_vertices(dcded, get_faces()).save(conf_path('rebuild2.obj', 'tst'))
    Mesh().from_vertices(vts, get_faces()).save(conf_path('rebuild0.obj', 'tst'))

    canvas.close()


if __name__ == '__main__':
    tst()
