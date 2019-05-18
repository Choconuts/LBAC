import json, os


configure = 'win.json'


def find_dir_upwards(dir_name, iter=5):
    if os.path.exists(dir_name):
        return dir_name
    else:
        return find_dir_upwards(os.path.join('..', dir_name), iter - 1)


def conf_path(key, base="db"):
    path = conf_value(key)
    base_dir = get_base(base)
    if path is not None:
        path = os.path.join(base_dir, path)
        return path
    return conf_value(key + '_a')


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


def get_base(key):
    base_dir = conf_value(key)
    if base_dir is None:
        base_dir = find_dir_upwards(key, 5)
    return base_dir


conf_json = os.path.join(find_dir_upwards('conf'), configure)


if __name__ == '__main__':
    print(conf_json)
    print(conf_path('betas'))
