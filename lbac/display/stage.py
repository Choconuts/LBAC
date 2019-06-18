from lbac.display.virtual_fitting import *
from com.mesh.array_renderer import *
from com.mesh.simple_display import *

vf = None


def main():
    global vf
    if vf is None:
        vf = VirtualFitting()

    vf.beta = np.random.random((10)) * 2 - 1
    # vf.pose = np.random.random((24, 3)) * 0.1

    vf.update_cloth_only()
    vf.update_cloth_only()
    vf.update_cloth_only()


    def draw():
        msr.render()

    msr = MeshRenderer(vf.cloth, [0.8, 0.8, 0.8])
    set_display(draw)
    set_init(init_array_renderer)
    run_glut()

    vf.close()


if __name__ == '__main__':
    vf = VirtualFitting()
    main()