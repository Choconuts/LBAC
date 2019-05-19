from com.posture.smpl import *
from com.path_helper import *
from com.posture.imitator import translate_to_smpl_thetas, load_motion


class PoseTranslator:
    def load(self, json_file):
        self.meta = load_json(json_file)
        return self

    def poses_list(self):
        return []


class JsonTranslator(PoseTranslator):
    def poses_list(self):
        data = []
        for seq in self.meta:
            data.append(np.array(seq))
        return data


class AmcTranslator(PoseTranslator):

    def poses_list(self):
        max_length = -1
        if 'max_len' in self.meta:
            max_length = self.meta['max_len']
        base = self.meta['dir']
        if 'abs' not in self.meta:
            base = join(get_base(), base)

        data = []
        for sub in self.meta['gen']:
            for seq in self.meta['gen'][sub]:
                id = str(sub) + '_' + str(seq)
                asf = join(base, id, str(sub) + '.asf')
                amc = join(base, id, id + '.amc')

                motions = load_motion(amc)
                if max_length >= 0:
                    thetas = translate_to_smpl_thetas(asf, motions[0: max_length])
                else:
                    thetas = translate_to_smpl_thetas(asf, motions)
                data.append(np.array(thetas))

        return data




