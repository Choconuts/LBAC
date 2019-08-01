import time


class Timer:
    def __init__(self, prt=True):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.print = prt

    def tick(self, msg=''):
        res = time.time() - self.last_time
        if self.print:
            print('%.6f' % res, msg)
        self.last_time = time.time()
        return res

    def tock(self, msg=''):
        if self.print:
            print(msg, '%.6f' % (time.time() - self.last_time))
        return time.time() - self.last_time

from functools import wraps

def timing(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        f(*args, **kwargs)
        time_cost = time.time() - start
        print("Time cost: %f" % time_cost)

    return decorated


if __name__ == '__main__':
    @timing
    def x():
        for i in range(1000000):
            f = 12 * 342 +1243

    x()