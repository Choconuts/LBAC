from com.path_helper import *



def find_all_valids_seqs(sim_dir, min_frame_thr=40, lost_frame_thr=40, offset=0, stride=1):
    assert stride > 0
    valids = []

    def check_id(i):
        return (i - offset) % stride == 0

    last_i = find_last_in_dir(sim_dir, lambda i: str5(i))
    for i in range(last_i):
        if not check_id(i):
            continue
        out = join(sim_dir, str5(i))
        last_f = find_last_in_dir(out, lambda i: str4(i) + '_00.obj')
        if min_frame_thr >= 0 and last_f > min_frame_thr:
            valids.append(i)

    save_json(valids, conf_path('temp/valid-seqs.json'))


if __name__ == '__main__':
    find_all_valids_seqs(conf_path('sim/tmp'), 1)