from lbac.display.stage import *



def tes_num():
    cloth = Mesh().load('../ctmp.obj')
    mesh = Mesh().load('../btmp.obj')
    offset = [np.array([0., 0.03, 0.005])]
    mesh.vertices += offset[0]

    def key(x, y, z):
        unit = 0.005
        if x == b'w':
            offset[0] += [0, 0, unit]
            mesh.vertices += [0, 0, unit]
        if x == b's':
            offset[0] += [0, 0, -unit]
            mesh.vertices += [0, 0, -unit]
        if x == b'a':
            offset[0] += [0, -unit, 0]
            mesh.vertices += [0, -unit, 0]
        if x == b'd':
            offset[0] += [0, unit, 0]
            mesh.vertices += [0, unit, 0]
        print(offset[0])
        glutPostRedisplay()

    view_meshes([mesh, cloth], None, key)


if __name__ == '__main__':
    tes_num()