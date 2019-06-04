
type_size = {
    'f': 4,
    'i': 4,
    'd': 8
}

protocol = {
    'v': [0, 7366 * 3, 'f'],
    'f': [0, 14496 * 3, 'i'],
    'uv': [1, 7366 * 2, 'f'],
    'p': [0, 24 * 3, 'f'],
    'b': [1, 4, 'f'],
    'i': [2, 1, 'i']
}


def total_size(attrs):
    tot = 0
    for attr in attrs:
        tot += protocol[attr][1] * type_size[protocol[attr][2]]
    return tot


