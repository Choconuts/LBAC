import json
import os
import copy


def py_path(file__):
    return os.path.split(os.path.realpath(file__))[0]

template_file = os.path.join(py_path(__file__), 'template.json')
arcsim_exe = '..\\..\\..\\..\ARCSim\\arcsim-0.21\\x64\Release\\adaptiveCloth-2.0.exe'
cloth_mesh = 'meshes/tshirt7.obj'
material = 'materials/gray-interlock.json'
shape_dir = '../../sequence/shape'
pose_dir = '../../sequence/pose'
out_dir = ['outputs\\shape', 'outputs\\pose']


def conf_tmp(i):
    return './conf/' + str(i) + '.json'


def out_path(t, i):
    return os.path.join(out_dir[t], str(i))

def sim(conf, out='', op=0):
    ops = ['simulate', 'simulateoffline']
    print( arcsim_exe + ' ' + ops[op] + ' ' + conf + ' ' + out)
    return arcsim_exe + ' ' + ops[op] + ' ' + conf + ' ' + out


def run_seq(seq_dir, i, option):
    with open(template_file, 'r') as fp:
        conf = json.load(fp)
    conf['cloths'][0]['mesh'] = cloth_mesh
    conf['cloths'][0]['materials'][0]['data'] = material

    def get(key, alt):
        if key in option:
            return option[key]
        else:
            return alt

    conf['end_frame'] = get('end_frame', 30)
    mo = conf['morph_obstacle']
    mo['frame'] = get('obs_frame', 5)
    mo['frame_time'] = get('frame_time', 0.033)
    mo['dir'] = os.path.join(seq_dir, 'seq_' + str(i))

    with open(conf_tmp(i), 'w') as fp:
        json.dump(conf, fp)

    k = get('type', 0)
    os.system(sim(conf_tmp(i), out_path(k, i), 1))


if __name__ == '__main__':
    for i in range(10, 20):
        run_seq(pose_dir, i, {'type': 1, 'end_frame': 120, 'obs_frame': 120})
    for i in range(17):
        run_seq(shape_dir, i, {'type': 0, 'end_frame': 30, 'obs_frame': 5})


