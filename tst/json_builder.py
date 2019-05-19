import json, os
from com.path_helper import *



x = {
    'dir': 'raw/mocap/1',
    'gen': {
        13: [29]
    },
    'max_len': 2331
}


if __name__ == '__main__':
    save_json(x, 'out.json')