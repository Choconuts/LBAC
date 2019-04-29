import os, shutil

def extract_shape_obj(shape_path, outpath):
    for i in range(17):
        f = os.path.join(shape_path, str(i), '0030_00.obj')
        if i == 12:
            f = os.path.join(shape_path, str(i), '0017_00.obj')
        shutil.copyfile(f, os.path.join(outpath, str(i) + '.obj'))

extract_shape_obj(r'E:\Choconuts\LBAC - 副本\LBAC-master\app\test\simulation\outputs\shape',
                  r'E:\Choconuts\LBAC - 副本\LBAC-master\app\test\simulation\outputs\shape_y_final')

if __name__ == '__main__':
    pass