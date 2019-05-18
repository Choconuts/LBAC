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

