import re
import numpy as np
import json

meta = {
    0: [0, 0, 0, 0, 0,  0, 0, 0, 0, 0],
}

for k in range(4):
    for i in range(4):
        maps = [-2, -1, 1, 2]
        tmp = [0, 0, 0, 0]
        tmp[k] = maps[i]
        for j in range(6):
            tmp.append(0)
        meta[k * 4 + i + 1] = tmp


def meta_file():
    print(meta)
    with open('betas.json', 'w') as fp:
        json.dump(meta, fp)



if __name__ == '__main__':
    pass
