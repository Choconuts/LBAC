import json, os

conf_json = '..\\..\\conf\\win.json'


def conf_path(key, base="db"):
    path = conf_value(key)
    if path is None:
        path = os.path.join(conf_value(base), conf_value(key + '_r'))
        return path
    return path


def conf_value(key):
    with open(conf_json, 'r') as fp:
        obj = json.load(fp)
        if key not in obj:
            return None
        return obj[key]


def str3(i):
    if i > 999:
        print("Warning: index overflowed")
    return '%03d' % i


def str4(i):
    if i > 9999:
        print("Warning: index overflowed")
    return '%04d' % i


def str5(i):
    if i > 99999:
        print("Warning: index overflowed")
    return '%05d' % i


if __name__ == '__main__':
    print(conf_path("betas_17"))
