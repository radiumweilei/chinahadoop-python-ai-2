import os
import time
from datetime import datetime
from multiprocessing import Process, Pool


def run_proc(n):
    print('第{}次循环，子进程id:{}，父进程id:{}'.format(n, os.getpid(), os.getppid()))
    time.sleep(1)


if __name__ == '__main__':

    print('父进程id', os.getpid())

    # 1. 顺序执行任务
    # start = datetime.now()
    # for i in range(10):
    #     run_proc(i)
    # print('耗时:', datetime.now() - start)

    # 2. 多进程并行执行
    # 2.1 多进程异步并行执行，进程间没有先后顺序
    # start = datetime.now()
    # for i in range(10):
    #     p = Process(target=run_proc, args=(i,))
    #     p.start()
    # print('耗时:', datetime.now() - start)

    # 2.2 多进程同步并行执行，进程间有先后顺序
    # start = datetime.now()
    # for i in range(10):
    #     p = Process(target=run_proc, args=(i,))
    #     p.start()
    # p.join()
    # print('耗时:', datetime.now() - start)

    # 3. 进程池管理多进程
    # 3.1 使用Pool管理多个进程，同步执行
    # pool = Pool()
    # start = datetime.now()
    # for i in range(10):
    #     pool.apply(func=run_proc, args=(i,))
    # pool.close()
    # pool.join()
    # print('耗时:', datetime.now() - start)

    # 3.2 使用Pool管理多个进程，异步执行
    # pool = Pool()
    # start = datetime.now()
    # for i in range(10):
    #     pool.apply_async(func=run_proc, args=(i,))
    # pool.close()
    # pool.join()
    # print('耗时:', datetime.now() - start)

