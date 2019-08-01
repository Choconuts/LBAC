from lbac.train import pose_gt as PoseGT
from lbac.display.seq_show import *
from com.sequence.sequence import *
from com.mesh.closest_vertex import VertexMapping
from lbac.display.stage import view_mesh, view_meshes
from lbac.test.projection_test import *


def gen_problem():
    workspace = conf_path('work', 'tst')
    ext_seq = '../../00091'
    # ext_seq = r'D:\Educate\CAD-CG\GitProjects\ext004adj1\00082'
    seq_meta, frame_num, beta, poses = parse_sim(ext_seq)

    meshes = []
    for i in range(60, 61):
        mesh_i = Mesh().load(join(ext_seq, str4(i) + '.obj'))
        meshes.append(mesh_i)

    return meshes, np.array(poses)[60:61], beta


def solve_problem():
    disp_seq = Sequence(0.033, "disp")
    disp_seq.data = []
    pose_gt.get_cloth_weights = get_weights
    for i in range(len(meshes)):
        disp = pose_gt.process(meshes[i], beta, poses[i])
        # disp = meshes[i].vertices - pose_gt.beta_gt.template.vertices
        disp_seq.data.append(disp)

    pose_seq = Sequence(0.066, "pose")
    pose_seq.data = poses
    # show_pose_seq_joints(pose_seq)
    show_seqs(pose_disp_seq=disp_seq,
              # pose_seq=pose_seq
              )


def get_weights(cloth: Mesh, body: Mesh, pose, smpl: SMPLModel, beta):
    cloth_weights = vertex_mapping.transfer(smpl.weights)
    return cloth_weights


if False:
    PoseGT.smpl = SMPLModel(conf_path('smpl'))
    smpl = PoseGT.smpl
    PoseGT.beta_gt.load(conf_path('gt/beta/1'))
    vertex_mapping = VertexMapping()
    meshes, poses, beta = gen_problem()
    smpl.set_params(pose=poses[0])
    posed_body = Mesh().from_vertices(smpl.verts, smpl.faces)
    # view_mesh(posed_body)

    vertex_mapping.to_mesh = meshes[0]
    vertex_mapping.from_mesh = posed_body
    vertex_mapping.calc()
    vertex_mapping.save(conf_path('temp_tst_map.json', 'tst'))
    # vertex_mapping.load(conf_path('temp_tst_map.json','tst'))
    # solve_problem()
    cloth = meshes[0]
    cloth.vertices = vertex_mapping.transfer(posed_body.vertices)
    # cloth_weights = vertex_mapping.transfer(smpl.weights)
    cloth.update_normal_only()
    view_mesh(cloth)
    # view_meshes([posed_body, cloth])


def model_link():
    meshes, poses, beta = gen_problem()
    cloth = meshes[0]
    smpl.set_params(pose=poses[0])
    posed_body = Mesh().from_vertices(smpl.verts, smpl.faces)

    rela = relation.get_rela()
    rela = relation.calc_rela_once(cloth, posed_body)
    lab = Lab()
    for i in range(len(cloth.vertices)):
        vc = cloth.vertices[i]
        vb = posed_body.vertices[rela[i]]
        lab.add_line(vc, vb)

    lab.graphic()


def weight_link():
    meshes, poses, beta = gen_problem()
    cloth0 = PoseGT.beta_gt.template
    body_0 = Mesh().from_vertices(smpl.verts, smpl.faces)
    posed_cloth = meshes[0]
    # smpl.set_params(pose=poses[0])
    # posed_body = Mesh().from_vertices(smpl.verts, smpl.faces)
    vertex_mapping = VertexMapping()
    vertex_mapping.to_mesh = cloth0
    vertex_mapping.from_mesh = body_0
    # vertex_mapping.calc()
    # vertex_mapping.save(conf_path('mapping.json', 'tst'))
    vertex_mapping.load(conf_path('mapping.json', 'tst'))
    cloth_verts = vertex_mapping.transfer(body_0.vertices)
    lab = Lab()
    for i in range(len(cloth0.vertices)):
        vc = cloth0.vertices[i]
        vb = cloth_verts[i]
        lab.add_line(vc, vb)
    lab.graphic()


def weight_gen():
    vertex_mapping = VertexMapping().load(conf_path('mapping.json', 'tst'))
    cloth0 = PoseGT.beta_gt.template
    body_0 = Mesh().from_vertices(smpl.verts, smpl.faces)
    vertex_mapping.to_mesh = cloth0
    vertex_mapping.from_mesh = body_0
    cloth_weights = vertex_mapping.transfer(smpl.weights)
    print(cloth_weights)
    save_json(cloth_weights, r'D:\Educate\CAD-CG\GitProjects\LBAC\db\model\relation\cloth_weights_7366.json')


def test_weight():
    cloth_weights = np.array(load_json(r'D:\Educate\CAD-CG\GitProjects\LBAC\db\model\relation\cloth_weights_7366.json'))
    print(cloth_weights.shape)
    meshes, poses, beta = gen_problem()
    posed_cloth = meshes[0]
    smpl.set_params(pose=poses[0])
    posed_body = Mesh().from_vertices(smpl.verts, smpl.faces)

    posed_cloth.vertices = dis_apply(smpl, cloth_weights, posed_cloth.vertices)
    view_mesh(posed_cloth)


if __name__ == '__main__':
    PoseGT.smpl = SMPLModel(conf_path('smpl'))
    smpl = PoseGT.smpl
    PoseGT.beta_gt.load(conf_path('gt/beta/1'))
    relation = ClosestVertex().load(conf_path('vert_rela'))
    test_weight()
