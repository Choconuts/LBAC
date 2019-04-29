import json
import os
import copy


def py_path(file__):
    return os.path.split(os.path.realpath(file__))[0]

# template_file = os.path.join(py_path(__file__), 'template.json')
# arcsim_exe = '..\\..\\..\\..\ARCSim\\arcsim-0.21\\x64\Release\\adaptiveCloth-2.0.exe'
# cloth_mesh = 'meshes/tshirt7.obj'
# material = 'materials/gray-interlock.json'
# shape_dir = '../../sequence/shape'
# pose_dir = '../../sequence/pose'
# out_dir = ['outputs\\shape', 'outputs\\pose']


class ARCSimRunner:

    def __init__(self,
                 template_conf,     # conf 的基本模板
                 arcsim_exe,        # arcsim的可执行程序的路径
                 cloth_mesh,        # 衣服模型的路径
                 material,          # 衣服的材料文件路径
                 shape_dir,         # shape的sequences, 路径下名称需为seq_x
                 pose_dir,          # pose的sequences, 路径下名称需为seq_x
                 out_dir):          # 输出的文件夹，文件夹会分pose和shape，分别从0计

        self.template_conf = template_conf
        self.arcsim_exe = arcsim_exe
        self.cloth_mesh = cloth_mesh
        self.material = material
        self.seq_dir = [shape_dir, pose_dir]
        self.out_dir = [out_dir + '\\shape', out_dir + '\\pose']
        if not os.path.isdir('conf_tmp'):
            os.mkdir('conf_tmp')

    def conf_tmp(self, i):
        return './conf_tmp/' + str(i) + '.json'

    def out_path(self, t, i):
        return os.path.join(self.out_dir[t], str(i))

    def sim(self, conf, out='', op=0):
        ops = ['simulate', 'simulateoffline']
        print( self.arcsim_exe + ' ' + ops[op] + ' ' + conf + ' ' + out)
        return self.arcsim_exe + ' ' + ops[op] + ' ' + conf + ' ' + out

    def run_seq(self, i, option):
        """

        :param seq_dir:
        :param i:
        :param option:
            type 0-shape, 1-pose
            end_frame 控制总帧数
            obs_frame 控制障碍物序列帧数，必要
            frame_time 控制帧率
            gravity 控制重力
            display 是否播放模拟
        :return:
        """
        with open(self.template_conf, 'r') as fp:
            conf = json.load(fp)
        conf['cloths'][0]['mesh'] = self.cloth_mesh
        conf['cloths'][0]['materials'][0]['data'] = self.material

        def get(key, alt):
            if key in option:
                return option[key]
            else:
                return alt

        k = get('type', 0)
        conf['end_frame'] = get('end_frame', 30)
        conf['gravity'] = get("gravity", conf['gravity'])
        mo = conf['morph_obstacle']
        mo['frame'] = get('obs_frame', 5)
        mo['frame_time'] = get('frame_time', 0.033)
        mo['dir'] = os.path.join(self.seq_dir[k], 'seq_' + str(i))

        with open(self.conf_tmp(i), 'w') as fp:
            json.dump(conf, fp)

        display = get('display', False)
        op = 1
        if display:
            op = 0

        os.system(self.sim(self.conf_tmp(i), self.out_path(k, i), op))


if __name__ == '__main__':
    pass


