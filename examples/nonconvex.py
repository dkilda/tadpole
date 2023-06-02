#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

import numpy   as np
import tadpole as td
import scipy.optimize

import timeit
import cProfile


'''
CPUTIME:  0.6949000549939228
         3044737 function calls (2875644 primitive calls) in 1.403 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   161574    0.070    0.000    0.110    0.000 itertools.py:272(__init__)
86520/65226    0.064    0.000    0.395    0.000 cache.py:22(wrap)
    99582    0.052    0.000    0.153    0.000 itertools.py:348(push)
    99582    0.049    0.000    0.085    0.000 itertools.py:293(_apply)
43344/23856    0.047    0.000    0.072    0.000 node.py:473(_node_by_type)
    56490    0.045    0.000    0.102    0.000 itertools.py:246(apply)
    56490    0.044    0.000    0.169    0.000 itertools.py:284(_list)
   294655    0.041    0.000    0.073    0.000 {built-in method builtins.isinstance}
250698/126882    0.041    0.000    0.304    0.000 {built-in method builtins.iter}
    28770    0.036    0.000    0.177    0.000 graph.py:364(attach)
    71316    0.034    0.000    0.279    0.000 itertools.py:323(__iter__)
    46242    0.032    0.000    0.094    0.000 graph.py:323(__init__)
    87193    0.031    0.000    0.031    0.000 {built-in method builtins.getattr}
    17472    0.030    0.000    0.305    0.000 graph.py:308(train_args)
     9702    0.028    0.000    0.348    0.000 graph.py:291(denodify)
    85680    0.023    0.000    0.031    0.000 itertools.py:391(execute)
    61992    0.022    0.000    0.027    0.000 itertools.py:232(__init__)
    56490    0.021    0.000    0.026    0.000 itertools.py:257(restore)
    46242    0.021    0.000    0.164    0.000 graph.py:274(_nodify)
    23394    0.020    0.000    0.112    0.000 node.py:513(point)
    23856    0.019    0.000    0.091    0.000 node.py:471(_create)
    17472    0.018    0.000    0.182    0.000 graph.py:272(nodify)
    65185    0.017    0.000    0.017    0.000 {built-in method _abc._abc_instancecheck}
     9702    0.017    0.000    0.468    0.000 graph.py:243(_denodified)
    99582    0.015    0.000    0.015    0.000 itertools.py:386(__init__)
     4410    0.015    0.000    1.052    0.000 graph.py:651(apply)

'''


x = np.array([0.0, 0.0])
i = td.IndexLit("i", 2)


def fun(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2 


def gradfun(x):
    inp = td.TensorGen(x, (i,))
    out = td.evaluate_with_gradient(fun)(inp)
    return td.asdata(out[0]), td.asdata(out[1])


def main():
    return scipy.optimize.minimize(gradfun, x, jac=True, method='CG')


print("minimum: ", main().x)

    

cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

cProfile.run('main()', sort='tottime')

"""
"""






