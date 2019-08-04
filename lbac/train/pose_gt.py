from com.learning.ground_truth import *
from com.path_helper import *
from com.mesh.smooth import *
from lbac.train.shape_gt import BetaGroundTruth, parse_sim, parse_ext
from com.mesh.closest_vertex import ClosestVertex
from com.posture.smpl import *

loaded_cloth_weights = None


def get_cloth_weights(cloth: Mesh, body: Mesh, pose, smpl, beta):
    global loaded_cloth_weights
    if loaded_cloth_weights is None:
        loaded_cloth_weights = np.array(load_json(conf_path(r'model/relation/cloth_weights_7366.json')))
    return loaded_cloth_weights


beta_gt = BetaGroundTruth()
relation = ClosestVertex()
smpl = None
smooth_times = 0


def gen_pose_gt_data(ext_dir, beta_dir, gt_dir, gen_range=None, smooth=0):
    global smpl, smooth_times
    smooth_times = smooth
    smpl = SMPLModel(conf_path('smpl'))
    meta, valid_dict = parse_ext(ext_dir)

    sim_type = meta['config']['type']
    if gt_dir is None:
        gt_dir = join(conf_path('temp'), sim_type)

    beta_gt.load(beta_dir)

    # 如果不划定范围，就从extract meta里读取所有有效序列（帧数大于插值数的
    if gen_range is None:
        gen_range = []
        for seq_idx in valid_dict:
            gen_range.append(seq_idx)

    valids = valid_dict.keys()

    gen_range = list(set(gen_range).intersection(set(valids)))

    if not exists(gt_dir):
        os.makedirs(gt_dir)

    # 从合法的列表里筛选
    meta['index'] = dict()
    for seq_idx in gen_range:
        frames = valid_dict[seq_idx]
        seq_idx = int(seq_idx)
        try:
            # 每个序列的meta信息（包括bp数据
            seq_meta, frame_num, beta, poses = parse_sim(join(ext_dir, str5(seq_idx)))
        except Exception as e:
            print(seq_idx, e)
            continue
        print(seq_idx)
        if frames != frame_num:
            # 最终使用的还是parse出来的frame num
            print('warning: invalid seq frames record')
            frames = frame_num
        disps = []
        # 生成disp数据
        for i in range(len(poses)):
            mesh_i = Mesh().load(join(ext_dir, str5(seq_idx), str4(i) + '.obj'))
            print(frames, len(poses))
            disp = process(mesh_i, beta, poses[i])
            disps.append(disp)
        meta['index'][seq_idx] = len(disps)

        # 把这个序列的所有有效数据合起来，保存成一个json，方便按序列读取
        data = dict()
        data['disps'] = np.array(disps).tolist()
        data['poses'] = poses
        data['beta'] = beta
        print(join(gt_dir, str5(seq_idx) + '.json'))
        save_json(data, join(gt_dir, str5(seq_idx) + '.json'))

    # pose gt的meta
    save_json(meta, join(gt_dir, 'meta.json'))


def process(mesh, beta, pose):
    """
    获取姿势模型的displacement
    :param mesh:
    :param beta:
    :param pose:
    :return:
    """
    # beta处理
    beta_a = np.array(beta)
    beta_full = np.hstack((beta_a, np.zeros(6)))

    # pose消除（目前直接读取保存的weights
    body = Mesh().from_vertices(smpl.verts, smpl.faces)
    cloth_weights = get_cloth_weights(cloth=mesh, body=body, beta=beta, pose=pose, smpl=smpl)
    smpl.set_params(pose=np.array(pose), beta=beta_full)    # 之前忘记设置pose和beta了
    mesh.vertices = dis_apply(smpl, cloth_weights, mesh.vertices)
    mesh.update()

    # # 把smooth改到了还原姿势的后面，比较鲁棒
    # if smooth_times > 0:
    #     smooth(mesh, smooth_times)

    disp = mesh.vertices - beta_gt.template.vertices

    beta_id = -1
    for i in beta_gt.data['betas']:
        beta_i = np.array(beta_gt.data['betas'][i])
        if (beta_a == beta_i).all():
            beta_id = int(i)
            break

    if beta_id < 0:
        print('warning: missing beta gt')
        return None

    # beta disp消除
    beta_disp = beta_gt.data['disps'][str(beta_id)]

    return disp - beta_disp


class PoseSampleId(SampleId):

    def derefer(self):
        coords = self.data[0][self.id]
        data = self.data[1][coords[0]]
        step = self.data[2]
        stride = self.data[3]

        p = coords[1]
        beta = np.zeros((step, 1)) + np.array(data['beta']).reshape((1, -1))
        poses = np.array(data['poses'][p:p+step]).reshape((step, -1))
        disps = np.array(data['disps'][p:p+step]).reshape((step, -1))

        # stride
        poses = []
        for i in range(step):
            poses.append(data['poses'][p + i * stride])
        poses = np.array(poses).reshape((step, -1))
        disps = []
        for i in range(step):
            disps.append(data['disps'][p + i * stride])
        disps = np.array(disps).reshape((step, -1))

        x = np.hstack((poses, beta))

        return x, disps


class PoseGroundTruth(GroundTruth):
    """
    structure:
    [{ id: (seqs, different length)
        {
         disps
         poses
         beta}
    }...]

    dereference method:

    """
    def __init__(self, step=5, cut_off=10, stride=1):
        self.step = step
        self.mapping = []
        self.test_cut = cut_off
        self.stride = stride

    def load(self, gt_dir):
        # self.meta = load_json(join(gt_dir, 'meta.json'))
        # index = self.meta['index']
        # new_index = dict()
        # total = 0
        # for i in index:
        #     if index[i] < self.step:
        #         # 序列的帧数小于step，舍弃
        #         continue
        #     # 序列的有效数量不再是帧数，而是能够向后取出step个帧的起始帧数目
        #     # 修改后应该可以往后取出跳过stride的共计step的数目
        #     new_index[i] = index[i] - self.step * self.stride + 1 * self.stride
        #     total += new_index[i]
        #     for j in range(new_index[i]):
        #         self.mapping.append((i, j))
        # max_num = total
        # self.batch_manager = BatchManager(max_num, max_num - self.test_cut)
        # self.samples = []
        # self.index = new_index

        self.load_meta(gt_dir)

        self.load_data(gt_dir)

        self.samples = []
        data = (self.mapping, self.data, self.step, self.stride)
        for i in range(len(self.mapping)):
            sample = PoseSampleId(i, data)
            self.samples.append(sample)

        return self

    def load_meta(self, gt_dir):
        self.meta = load_json(join(gt_dir, 'meta.json'))
        index = self.meta['index']
        new_index = dict()
        # total是全部的X帧切片数量
        total = 0
        for i in index:
            if index[i] < self.step:
                # 序列的帧数小于step，舍弃
                continue
            # 序列的有效数量不再是帧数，而是能够向后取出step个帧的起始帧数目
            # 修改后应该可以往后取出跳过stride的共计step的数目
            new_index[i] = index[i] - self.step * self.stride + 1 * self.stride
            total += new_index[i]
            for j in range(new_index[i]):
                self.mapping.append((i, j))
        print("total sample:", total)
        max_num = total
        self.batch_manager = BatchManager(max_num, max_num - self.test_cut)
        self.index = new_index

    def load_data(self, gt_dir):
        self.data = dict()
        for i in self.index:
            self.data[i] = load_json(join(gt_dir, str5(int(i)) + '.json'))

    def get_batch(self, size):
        last = self.batch_manager.pointer
        ids = self.batch_manager.get_batch(size)
        if last > self.batch_manager.pointer:
            self.batch_manager.shuffle()
        batch = [[], []]
        for id in ids:
            sample = self.samples[id].derefer()
            batch[0].append(sample[0])
            batch[1].append(sample[1])
        return batch

    def get_test(self):
        ids = self.batch_manager.get_test()

        batch = [[], []]
        for id in ids:
            sample = self.samples[id].derefer()
            batch[0].append(sample[0])
            batch[1].append(sample[1])
        return batch


if __name__ == '__main__':
    a = [1, 2, 2]
    b = [2, 3, 5]
    print(list(set(a).intersection(set(b))))