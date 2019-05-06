import json
import os
import copy

template_file = 'template.json'


def py_path(file__):
    return os.path.split(os.path.realpath(file__))[0]


class Configure:
    # fps = 30
    steps_per_frame = 10
    total_frames = 40
    frame_time = 0.033

    template = os.path.join(py_path(__file__), template_file)

    def build_simulation(self, conf_dir, cloth_path, obstacle_path):
        with open(template_file, 'r') as fp:
            conf = json.load(fp)
            conf['cloths'][0]['mesh'] = cloth_path
            conf['end_frame'] = self.total_frames
            conf['frame_time'] = self.frame_time
            conf['frame_steps'] = self.steps_per_frame

            if not os.path.isfile(obstacle_path):
                sequence = get_sequence_list(obstacle_path)
                obs_template = conf["obstacles"][0]
                conf["obstacles"] = []
                time = 0
                step = self.frame_time / self.steps_per_frame
                for obs_file in sequence:
                    obstacle = copy.deepcopy(obs_template)
                    obstacle["mesh"] = obs_file
                    obstacle["start_time"] = time
                    time += step
                    obstacle["end_time"] = time
                    conf["obstacles"].append(obstacle)
                conf["obstacles"][len(conf["obstacles"]) - 1]["end_time"] = 10
            else:
                conf["obstacles"][0]['mesh'] = obstacle_path
                conf["obstacles"][0]['start_time'] = 0
        with open(conf_dir, 'w') as fp:
            json.dump(conf, fp)
        return self



def get_sequence_list(seq_dir):
    seq = []
    i = -1
    files = os.listdir(seq_dir)
    if len(files) < 0:
        return []
    file = '.obj'
    while i < 0 or file in files:
        if i >= 0:
            seq.append(seq_dir + '/' + file)
        i += 1
        file = str(i) + '.obj'
    print(seq)
    return seq


if __name__ == '__main__':
    Configure().build_simulation('../conf-seq.json', 'meshes/square4.obj', '../anima/seq1')