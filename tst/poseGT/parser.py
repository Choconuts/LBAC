from com.path_helper import *


def parse_pose_gt(pose_gt_dir):
    meta = load_json(join(pose_gt_dir, 'meta.json'))
    index = meta['index']
    seqs = []
    for si in index:
        if index[si] == 101:
            seqs.append(int(si))

    """
    去除所有同beta的
    """

    no_same_beta_seqs = dict()
    for s in seqs:
        no_same_beta_seqs[int(s / 17)] = s

    no_same_beta_seqs = list(no_same_beta_seqs.values())

    print(len(no_same_beta_seqs))
    print(no_same_beta_seqs)

    copy_gt(pose_gt_dir, no_same_beta_seqs, 'result')


def copy_gt(pose_gt_dir, copy_range, dst_dir):
    meta = load_json(join(pose_gt_dir, 'meta.json'))
    index = meta['index']

    new_index = dict()
    for i in copy_range:
        new_index[i] = index[str(i)]

    meta['index'] = new_index

    print(meta)

    import shutil

    for i in copy_range:
        print(join(pose_gt_dir, str5(i) + '.json'), join(dst_dir, str5(i) + '.json'))
        shutil.copy(join(pose_gt_dir, str5(i) + '.json'), join(dst_dir, str5(i) + '.json'))

    save_json(meta, join(dst_dir, 'meta.json'))



