import numpy as np
import os
import json
from com.path_helper import *


class SeqReader:
    def __init__(self, seq_dir):
        self.seq_dir = seq_dir
        self.seq_metas = dict()
        i = 0
        while os.path.exists(os.path.join(self.seq_dir, 'seq_' + str5(i))):
            i += 1
        self.seq_num = i

    def load_meta(self, i):
        meta_file = os.path.join(self.seq_dir, 'seq_' + str5(i), 'meta.json')
        with open(meta_file, 'r') as fp:
            obj = json.load(fp)
        skip = 0
        if 'interp' in obj:
            skip = obj['interp']
        obj['poses'] = obj['poses'][skip:]
        self.seq_metas[i] = obj
        return obj


class SimExtractor:
    def __init__(self, sim_dir, seq_reader: SeqReader, config):
        self.sim_dir = sim_dir
        self.meta = dict()
        self.meta['sim_dir_info'] = sim_dir
        self.meta['valids'] = dict()
        self.meta['config'] = config

        self.extract_dir = ''
        self.seq_reader = seq_reader
        i = 0
        while os.path.exists(os.path.join(self.sim_dir, str5(i))):
            i += 1
        if 'sim_range' in config:
            self.ext_range = config['sim_range']
        with open(os.path.join(self.extract_dir, 'meta.json'), 'w') as fp:
            json.dump(self.meta, fp)

    def extract_seq(self, i, seq_meta):
        """
        把第i个序列的模拟结果的有效帧（不含插值）复制到目标文件夹内，序号减去插值数
        :param i:
        :return:
        """
        skip = 0
        if 'interp' in seq_meta:
            skip = seq_meta['interp']

        ext_dir = join(self.extract_dir, str5(i))

        sim_out = join(self.sim_dir, str5(i))
        import shutil
        if not exists(join(sim_out, 'conf.json')):
            print('sim %05d failed' % i)
            return 0
        if not exists(ext_dir):
            os.makedirs(ext_dir)
        shutil.copy(join(sim_out, 'conf.json'), join(ext_dir, 'conf.json'))
        frame = 0
        mesh = str4(frame) + '_00.obj'
        file_names = os.listdir(sim_out)
        valid_count = 0
        while mesh in file_names:
            shutil.copy(join(sim_out, mesh), join(ext_dir, str4(frame - skip - 1) + '.obj'))
            if frame > skip:
                valid_count += 1
            frame += 1
            mesh = str4(frame) + '_00.obj'
        if valid_count == 0:
            return valid_count
        seq_meta['seq_frames'] = seq_meta['frames']
        seq_meta['frames'] = valid_count
        seq_meta['poses'] = seq_meta['poses'][0:valid_count]

        save_json(seq_meta, join(ext_dir, 'meta.json'))

        return valid_count

    def extract(self, extract_dir):
        self.extract_dir = extract_dir
        ext_range = self.ext_range
        print('extract to %s, range: ' % extract_dir, ext_range)
        for i in ext_range:
            meta = self.seq_reader.load_meta(i)
            assert 'frames' in meta
            count = self.extract_seq(i, meta)
            if count > 0:
                print('%d' % count, end=' ')
                self.meta['valids'][i] = count
        with open(join(self.extract_dir, 'meta.json'), 'w') as fp:
            json.dump(self.meta, fp)


if __name__ == '__main__':
    print(str4(-2))