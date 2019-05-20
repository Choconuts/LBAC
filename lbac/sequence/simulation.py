import json
import os
import copy
from com.path_helper import *
from lbac.sequence.reader import SeqReader, SimExtractor
import time

template_file = conf_path('arcsim_conf_template')
arcsim_exe = conf_path('arcsim')
cloth_dir = conf_path('clothes')
material = conf_path('material')
temp_dir = conf_path('temp')

m_cloth_id = 0
m_out_dir = ''
m_sim_type = 1
m_seq_reader = None
m_sim_range = []
m_sim_time = 0


def cloth_mesh():
    return os.path.join(cloth_dir, str3(m_cloth_id) + '.obj')


def conf_tmp(i):
    conf_dir = os.path.join(temp_dir, 'conf')
    if not os.path.exists(conf_dir):
        os.makedirs(conf_dir)
    return os.path.join(conf_dir, str5(i) + '.json')


def out_path(i):
    return os.path.join(m_out_dir, str5(i))


def sim(conf, out='', op=0):
    ops = ['simulate', 'simulateoffline']
    print(arcsim_exe + ' ' + ops[op] + ' ' + conf + ' ' + out)
    return arcsim_exe + ' ' + ops[op] + ' ' + conf + ' ' + out


def get_seq_frame(i):
    return m_seq_reader.load_meta(i)['frames']


def run_seq(i, option):
    seq_dir = m_seq_reader.seq_dir
    with open(template_file, 'r') as fp:
        conf = json.load(fp)
    conf['cloths'][0]['mesh'] = cloth_mesh()
    conf['cloths'][0]['materials'][0]['data'] = material

    def get(key, alt):
        if key in option:
            return option[key]
        else:
            return alt

    conf['end_frame'] = get('end_frame', get_seq_frame(i))
    mo = conf['morph_obstacle']
    mo['frame'] = get('obs_frame', get_seq_frame(i))
    mo['frame_time'] = get('frame_time', 0.033)
    mo['dir'] = os.path.join(seq_dir, 'seq_' + str5(i))

    with open(conf_tmp(i), 'w') as fp:
        json.dump(conf, fp)

    os.system(sim(conf_tmp(i), out_path(i), m_sim_type))


def simulate(seq_reader: SeqReader, out_dir, cloth_id, sim_range, sim_mode=1, option=None):
    global m_out_dir, m_cloth_id, m_sim_type, m_seq_reader, m_sim_range, m_sim_time
    m_out_dir = out_dir
    m_cloth_id = cloth_id
    m_sim_type = sim_mode
    m_seq_reader = seq_reader
    m_sim_range = []
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if option is None:
        option = {}

    m_sim_time = time.time()
    for i in sim_range:
        m_sim_range.append(i)
        run_seq(i, option)

    # 秒为单位
    m_sim_time = time.time() - m_sim_time
    return True


def extract_results(extract_dir, sim_type, seq_reader=None):
    config = dict()
    config['cloth'] = m_cloth_id
    config['sim_range'] = m_sim_range
    config['type'] = sim_type
    config['time'] = m_sim_time
    if seq_reader is None:
        seq_reader = m_seq_reader
    extractor = SimExtractor(m_out_dir, seq_reader, config)
    extractor.extract(extract_dir)


if __name__ == '__main__':
    """
    """


