from com.mesh.mesh import *
from com.mesh.simple_display import *
from com.mesh.array_renderer import *
from com.sequence.sequence import *


props = ['time_step', 'frame_num', 'mesh_dict']


class ClothAnimation:

    def __init__(self, time_step=0.066):
        self.time_step = time_step
        self.mesh_dict = dict()
        self.frame_num = 0

    def set_mesh_list(self, verts_seq: Sequence, tag: str):
        verts_seq.re_sampling(self.time_step)
        self.mesh_dict[tag] = []
        if self.frame_num == 0:
            self.frame_num = verts_seq.get_frame_num()
        else:
            verts_seq.data = verts_seq.data[:self.frame_num]
        for verts in verts_seq.data:
            mesh = Mesh().from_vertices(verts, self.faces_dict[tag])
            mesh.update()
            self.mesh_dict[tag].append(mesh)

    def create_animate_mesh(self, mesh, tag, color=None):
        if color is None:
            color = [0.5, 0.5, 0.5]
        self.mesh_dict[tag] = {
            'faces': mesh.faces,
            'color': color
        }
        return self

    def set_mesh_sequence(self, verts_seq: Sequence, tag):
        mesh_info = self.mesh_dict[tag]
        verts_seq.re_sampling(self.time_step)
        mesh_info['verts'] = verts_seq.data
        if self.frame_num < verts_seq.get_frame_num():
            print("Error: expend frames", self.frame_num, verts_seq.get_frame_num())
        self.frame_num = verts_seq.get_frame_num()
        normals = []
        i = 0
        print('set', tag, 'verts, total', verts_seq.get_frame_num())
        for verts in verts_seq.data:
            print('frame', i)
            i += 1
            mesh = Mesh().from_vertices(verts, mesh_info['faces'])
            mesh.compute_vertex_normal()
            normals.append(mesh.normal)
        mesh_info['norms'] = np.array(normals)

    def save(self, anima_file):
        o = dict()
        for p in props:
            v = getattr(self, p)
            o[p] = v
        save_json(o, anima_file)
        return self

    def load(self, anima_file):
        o = load_json(anima_file)
        for p in props:
            setattr(self, p, o[p])
        for tag in self.mesh_dict:
            m = self.mesh_dict[tag]
            m['norms'] = np.array(m['norms'])
            m['verts'] = np.array(m['verts'])
            m['faces'] = np.array(m['faces'])
        return self

    def show(self):
        msrs = []
        mesh_dict = dict()
        for tag in self.mesh_dict:
            mesh = Mesh().from_vertices(self.mesh_dict[tag]['verts'][0], self.mesh_dict[tag]['faces'])
            msr = MeshRenderer(mesh, np.array(self.mesh_dict[tag]['color']))
            mesh_dict[tag] = mesh
            msrs.append(msr)

        def draw():
            for msr in msrs:
                msr.render()

        frm = [0, False]

        def idle():
            if frm[1]:
                return
            frm[0] += 1
            frm[0] %= self.frame_num
            try:
                mesh_dict[tag].vertices = self.mesh_dict[tag]['verts'][frm[0]]
                mesh_dict[tag].normal = self.mesh_dict[tag]['norms'][frm[0]]
            except:
                pass
            glutPostRedisplay()

        def key(k, xx, yy):
            if k == b' ':
                frm[1] = not frm[1]

        set_init(init_array_renderer)
        set_display(draw)
        set_callbacks(idle, key)
        run_glut()
        return self


