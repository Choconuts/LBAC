# encoding: utf-8
import mmap
import contextlib
import time
import numpy as np


connections = dict()
close_flags = dict()
start_time = dict()

class Connection:
    def __init__(self, id, size):
        if id in close_flags:
            print('warning: reopen shared memory')
        self.id = id
        self.mem_size = size
        close_flags[id] = True
        self.tick = 0
        self.idle = []
        self.init = []

    def start(self):
        for init in self.init:
            init()
        main_loop(self)
        return self

    def close(self):
        close_flags[self.id] = False
        return self


def main_loop(connection: Connection):
    if connection.id not in close_flags:
        print('id conflict!')
        return
    with contextlib.closing(mmap.mmap(-1, connection.mem_size, tagname=connection.id, access=mmap.ACCESS_WRITE)) as m:
        i = 0
        while close_flags[connection.id]:
            connection.tick = i
            for idle in connection.idle:
                idle(m)
            i += 1
    close_flags.pop(connection.id)


def create_shared_connect(size, mid, interval):
    """
    建立一个新的共享内存连接
    :param size: bytes num
    :param mid: memory id
    :param interval: fixed interval between idle step
    :return: connect
    """
    if mid in connections:
        print("error, reopen same memory")
        return None

    connection = Connection(mid, size)

    def init():
        start_time[connection.id] = time.time()

    def idle(m):
        t = time.time()
        st = start_time[connection.id]
        i = connection.tick + 1
        if t - st < i * interval:
            time.sleep(i * interval - t + st)

    connection.init.append(init)
    connection.idle.append(idle)

    return connection


def close_shared_connect(connection):
    """
    关闭一个共享内存连接
    :param cid:
    :return:
    """
    connection.close()




