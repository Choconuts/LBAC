import sshtunnel
import os, sys

model_name = 'lbac'
dir_name = 'train'
file_name = 'pose_gt.py'


if __name__ == '__main__':
    os.system('scp ..\\'+ model_name + '\\' + dir_name + '\\' + file_name + ' ChenYanzhen@10.76.2.238:Projects/LBAC2/'+ model_name + '/' + dir_name + '/' + file_name)
    os.system('cyzzz123')