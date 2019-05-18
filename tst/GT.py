import numpy as np


class Groundtruth:

    def __init__(self):
        self.element = np.linspace(0, 128)





if __name__ == '__main__':
    import json
    param = [-2, -1, 1, 2]

    out = [np.zeros(4).tolist()]

    for i in range(4):
        for j in range(4):
            beta = np.zeros(4)
            beta[i] = param[j]
            out.append(beta.tolist())

    with open('betas_17.json', 'w') as fp:
        json.dump(out, fp)
