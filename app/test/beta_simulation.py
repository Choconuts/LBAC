import json
import os

base_dir = os.getcwd()
arcsim_dir = '../'
bodies_dir = './'
conf_dir = 'conf'
output_dir = 'out'


def edit_conf(cloth_path, obs_path, end_time, i):
    with open(base_dir + '/tshirt-template.json', 'r') as fp:
        conf = json.load(fp)
        conf['cloths'][0]['mesh'] = cloth_path
        conf["obstacles"][0]['mesh'] = obs_path
        conf['end_time'] = end_time
    with open(conf_dir + '/' + str(i) + '.json', 'w') as fp:
        json.dump(conf, fp)


def prepare_simulation():
    for i in range(17):
        obs = bodies_dir + '/' + str(i) + '.obj'
        edit_conf('', obs, 0.8, i)
    os.chdir(arcsim_dir)


def run_simulation():
    for i in range(17):
        config = conf_dir + '/' + str(i) + '.json'
        cmd = r'x64\release\arcsim simulateoffline ' + config + ' ' + output_dir
        os.system(cmd)


if __name__ == '__main__':
    prepare_simulation()
    run_simulation()