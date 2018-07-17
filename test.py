from multiprocessing import Process, Manager
import numpy as np

def f(d, l):
    d[1] = np.zeros(shape=(100,200))
    d['2'] = 2
    d[0.25] = None
    l.reverse()

if __name__ == '__main__':
    manager = Manager()

    d = manager.dict()
    l = manager.list(range(10))

    p = Process(target=f, args=(d, l))
    p.start()
    p.join()

    print (d[1].shape)
    print (l)
    
    a = True
    a = not a
    print(a)