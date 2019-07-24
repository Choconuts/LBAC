from tst import a


a.x = 3
#
# if __name__ == '__main__':
#     import os
#     print(os.listdir('.'))




import numpy as np



class A:

    def __init__(self, l):
        self.l = l




if __name__ == '__main__':
    a = []
    b = [1, 2, 3]
    for i in range(10):
        a.append(A(b))

    a[0].l[1] = 10
    print(a[1].l)