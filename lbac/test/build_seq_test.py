from lbac.display.seq_show import *
from lbac.sequence.builder import *
from lbac.display.cloth_animation import *
from threading import Thread


def show_model_sequence(seq_dir, key):
    cache = []

    def loading():
        i = 0
        path = join(seq_dir, key(i))
        while exists(path):
            print('load:', i)
            cache.append(Mesh().load(path))
            path = join(seq_dir, key(i))
            i += 1

    def display():
        while len(cache) < 1:
            time.sleep(0.1)
        msr = MeshRenderer(cache[0], [0.6, 0.6, 0.6])
        def draw():
            msr.render()

        frm = [0, False]

        def idle():
            if frm[1]:
                return
            frm[0] += 1
            frm[0] %= len(cache)
            try:
                msr.mesh = cache[frm[0]]
            except:
                pass
            glutPostRedisplay()

        def key(k, xx, yy):
            if k == b' ':
                frm[1] = not frm[1]

        set_init(init_array_renderer)
        set_display(draw)
        set_callbacks(idle, key)
        run_glut()

    Thread(target=loading).start()
    display()


if __name__ == '__main__':
    show_model_sequence('../../../adj_seqs/seq_00045', lambda i: str(i) + '.obj')