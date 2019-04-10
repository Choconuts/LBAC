from app.display.objects import OBJ
from app.geometry.mesh import Mesh
import os
from app.test.simulation.conf_builder import py_path


def seq_builder():
    obj = Mesh().load('sphere.obj')
    # os.mkdir('seq1')
    obj.vertices *= 0.4
    for i in range(300):
        obj.vertices += [0, 0, 0.003]
        obj.save('seq1/' + str(i) + '.obj')


if __name__ == '__main__':
    seq_builder()
    print(py_path(__file__))