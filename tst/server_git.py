import cv2
import numpy as np
import math

def save_pic(mat):
    cv2.imwrite('sphere64.jpg', mat)

def gen_sphere(a):
    img = np.zeros((a, a, 3), np.uint8)
    c = a / 2

    for x in range(a):
        for y in range(a):
            r2 = 1 - (x/c - 1)**2 - (y/c - 1)**2
            if r2 < 0:
                continue
            img[x, y] = int(math.sqrt(r2) * 255)

    cv2.imshow('s', img)
    cv2.waitKey(0)
    return img

if __name__ == '__main__':
    # img = gen_sphere(64)
    # save_pic(img)
    """
    """
    eps = 0.15
    x = math.exp(-1/2 * (0.1 / eps)**2)
    print(x)
    x = math.exp(-1/2 * (0.2 / eps)**2)
    print(x)
    x = math.exp(-1/2 * (0.3 / eps)**2)
    print(x)
    x = math.exp(-1/2 * (0.4 / eps)**2)
    print(x)