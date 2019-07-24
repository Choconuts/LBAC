import numpy as np
from com.path_helper import *
import math


class Sequence:
    def __init__(self, time_step, typ=''):
       self.time_step = time_step
       self.type = typ
       self.meta = dict()
       self.data = np.array([])

    @staticmethod
    def frame_name(i):
        return str5(i)

    def get_frame_num(self):
        return len(self.data)

    def get_total_time(self):
        return self.get_frame_num() * self.time_step

    def save(self, path, save_as_dir=False):
        if not save_as_dir:
            obj = dict()
            obj['time_step'] = self.time_step
            obj['type'] = self.type
            obj['meta'] = self.meta
            obj['data'] = self.data.tolist()
            save_json(obj, path)
        else:
            if not exists(path):
                os.makedirs(path)
            meta = dict()
            meta['time_step'] = self.time_step
            meta['type'] = self.type
            meta['meta'] = self.meta
            save_json(meta, join(path, 'meta.json'))
            for i in range(len(self.data)):
                frame = self.data[i]
                if type(frame) == np.ndarray:
                    frame = frame.tolist()
                save_json(frame, join(path, self.frame_name(i) + '.json'))

    def load(self, path, load_as_dir=False):
        if not exists(path):
            return
        if os.path.isdir(path):
            load_as_dir = True
        if not load_as_dir:
            obj = load_json(path)
            self.time_step = obj['time_step']
            self.type = obj['type']
            self.meta = obj['meta']
            self.data = np.array(obj['data'])
        else:
            meta = load_json(join(path, 'meta.json'))
            self.time_step = meta['time_step']
            self.type = meta['type']
            self.meta = meta['meta']
            last_i = find_last_in_dir(path, lambda i: self.frame_name(i) + '.json')
            self.data = []
            for i in range(last_i):
                frame = load_json(join(path, self.frame_name(i) + '.json'))
                if type(frame) == list:
                    frame = np.array(frame)

                self.data.append(frame)
            self.data = np.array(self.data)

    def get_shot_at(self, time):
        index = math.floor(time / self.time_step)
        if index == len(self.data) - 1:
            return self.data[index]
        offset = time % self.time_step
        t = offset / self.time_step
        return self.data[index] * (1 - t) + self.data[index + 1] * t

    def get_frame_rate(self):
        return 1 / self.time_step

    def set_frame_rate(self, rate):
        self.time_step = 1 / rate

    def down_sampling(self, times):
        raise Exception("not implement")



