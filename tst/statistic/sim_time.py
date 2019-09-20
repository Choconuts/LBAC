from com.path_helper import *
import numpy as np
import re


def extract_str(s):
    return re.match('>sec', s)


def line_parse(file, parse):
    with open(file, 'r') as fp:
        ss = fp.readlines()
        for i in range(len(ss)):
            parse(ss[i], i)


def change(file):
    il = [0]
    times = []
    def parse(s, ii):
        res = re.match('>sec ([0-9.-]*)', s)
        if res:
            il[0] += 1
            g = res.groups()
            id = str(il[0])
            if id != g[0][:len(id)]:
                il[0] = 0
                return
            t = g[0][len(id):]
            # print(g[0][:len(id)], s, t)
            if il[0] > 1:
                times.append(float(t))
    line_parse(file, parse)
    print(np.mean(times) * 100000 / 3600 / 24)


def calc_days_needed(file):
    il = [0]
    times = []
    def parse(s, ii):
        res = re.match('>sec ([0-9.-]+) ([0-9.-]+)', s)
        if res:
            t = res.groups()[1]
            print(t)
            times.append(float(t))
    line_parse(file, parse)
    print(np.mean(times) * 100000 / 3600 / 24)


if __name__ == '__main__':
    # change(r'D:\Educate\CAD-CG\GitProjects\adj_106-.out')
    calc_days_needed(r'D:\Educate\CAD-CG\GitProjects\adj_112.out')