from com.path_helper import *
from tst.workspace.seq_cut import produce_poses

odd_path = r'D:\Educate\CAD-CG\GitProjects\LBAC\tst\workspace\valid-odd.json'
in_path = r'D:\Educate\CAD-CG\GitProjects\seqs_128_r.json'
even_path = r'D:\Educate\CAD-CG\GitProjects\LBAC\tst\workspace\valid-even.json'
all_128_path = r'D:\Educate\CAD-CG\GitProjects\LBAC\tst\workspace\valid-all.json'
out_path = r'D:\Educate\CAD-CG\GitProjects\LBAC\tst\workspace\seqs_129_a.json'


def check_odd_valid():
    valid = load_json(odd_path)
    print(valid.__len__())
    print(valid)


def gen_even_valid():
    v = []
    for i in range(56):
        v.append(i * 2)
    for i in range(16):
        v.append(i + 112)
    save_json(v, 'valid-even.json')


def gen_all_128_valids():
    odd = load_json(odd_path)
    even = load_json(even_path)
    odd.extend(even)
    print(len(odd))
    save_json(odd, 'valid-all.json')


def gen_even_poses_json():
    produce_poses(
        in_file=in_path,
        out_file=out_path,
        valid_file=all_128_path,
        add_zero_flag=True
    )


def check_129():
    poses = load_json(out_path)
    print(len(poses))
    for p in poses:
        p = np.array(p)
        print(p.shape)


if __name__ == '__main__':
    # check_odd_valid()
    # gen_even_valid()
    check_129()


