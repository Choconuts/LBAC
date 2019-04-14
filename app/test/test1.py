import re
import numpy as np
from scipy import interpolate
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


xp = [1, 2, 3]
fp = [3, 2, 0]

x = [0, 1, 2, 3, 4, 5]

# interpolate.interp2d(x, y, fvals, kind='cubic')

b1 = [0, 0, 0, 0]
b2 = [2, 3, 4, 5]
r = []
for i in range(len(b1)):
    r.append(np.linspace(b1[i], b2[i], 10))
r = np.transpose(r)
print(r)

if __name__ == '__main__':
    pass
