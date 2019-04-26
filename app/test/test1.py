import re
import numpy as np
from scipy import interpolate
import json
from pyoctree import pyoctree as ot

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

vts = []
for i in [-1, 1]:
    for j in [-1, 1]:
        for k in [-1, 1]:
            vts.append([i, j, k])
for i in [-1, 1]:
    for j in [-1, 1]:
        for k in [-1, 1]:
            vts.append([i * 0.5, j * 0.5, k * 0.5])
for i in [-1, 1]:
    for j in [-1, 1]:
        for k in [-1, 1]:
            vts.append([i * 4, j * 0.5, k * 0.5])
for i in [-1, 1]:
    for j in [-1, 1]:
        for k in [-1, 1]:
            vts.append([i * 0.5, j * 0.8, k * 5])
print(vts)
vts = np.array(vts, np.float)

fcs = []
for i in range(30):
    fcs.append([i, i, i])
for i in range(16):
    fcs.append([i, i, i])
for i in range(30):
    fcs.append([i, i, i])
for i in range(16):
    fcs.append([i, i, i])

oct =  ot.PyOctree(vts, np.array(fcs, np.int32))

print(oct.root)
print(oct.root.branches)

if __name__ == '__main__':
    pass
