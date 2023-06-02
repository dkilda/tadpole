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
         6749467 function calls (6381015 primitive calls) in 3.065 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   355764    0.147    0.000    0.235    0.000 itertools.py:272(__init__)
195436/145460    0.138    0.000    0.858    0.000 cache.py:22(wrap)
   217990    0.112    0.000    0.336    0.000 itertools.py:348(push)
   217990    0.110    0.000    0.190    0.000 itertools.py:293(_apply)
   125026    0.099    0.000    0.359    0.000 itertools.py:284(_list)
   125026    0.097    0.000    0.215    0.000 itertools.py:246(apply)
82594/52466    0.092    0.000    0.142    0.000 node.py:473(_node_by_type)
   635141    0.087    0.000    0.153    0.000 {built-in method builtins.isinstance}
572522/293128    0.087    0.000    0.657    0.000 {built-in method builtins.iter}
    62966    0.075    0.000    0.385    0.000 graph.py:364(attach)
   162110    0.073    0.000    0.601    0.000 itertools.py:323(__iter__)
   198450    0.070    0.000    0.070    0.000 {built-in method builtins.getattr}
   102068    0.069    0.000    0.201    0.000 graph.py:323(__init__)
    21352    0.063    0.000    0.754    0.000 graph.py:291(denodify)
    51266    0.060    0.000    0.249    0.000 node.py:513(point)
    39102    0.056    0.000    0.651    0.000 graph.py:308(train_args)
   186774    0.050    0.000    0.069    0.000 itertools.py:391(execute)
   125026    0.048    0.000    0.049    0.000 itertools.py:257(restore)
   137774    0.048    0.000    0.057    0.000 itertools.py:232(__init__)
   102068    0.045    0.000    0.361    0.000 graph.py:274(_nodify)
    52466    0.045    0.000    0.186    0.000 node.py:471(_create)
    39102    0.041    0.000    0.402    0.000 graph.py:272(nodify)
    21352    0.037    0.000    1.031    0.000 graph.py:243(_denodified)
   134343    0.036    0.000    0.036    0.000 {built-in method _abc._abc_instancecheck}
   217990    0.034    0.000    0.034    0.000 itertools.py:386(__init__)
     9526    0.032    0.000    2.308    0.000 graph.py:651(apply)
   134343    0.030    0.000    0.066    0.000 abc.py:96(__instancecheck__)
   125026    0.028    0.000    0.045    0.000 itertools.py:241(__iter__)
    62966    0.028    0.000    0.413    0.000 node.py:432(concat)
    59276    0.027    0.000    0.195    0.000 graph.py:446(_layers)
    14126    0.026    0.000    0.227    0.000 graph.py:480(deshell)
    40576    0.026    0.000    0.226    0.000 graph.py:452(innermost)
    52576    0.025    0.000    0.222    0.000 graph.py:457(layer)
    17750    0.022    0.000    0.528    0.000 graph.py:248(concat)
    14150    0.021    0.000    0.486    0.000 graph.py:642(packs)

'''


x = td.TensorGen([[0.52, 1.12,  0.77],
                  [0.88, -1.08, 0.15],
                  [0.52, 0.06, -1.30],
                  [0.74, -2.49, 1.39]],
                 (td.IndexLit("i",4), td.IndexLit("j",3)))


targets = td.TensorGen([1, 1, 0, 1], (td.IndexLit("i",4),))


def lossfun(w):

    y = (td.tanh(x @ w) + 1) / 2
    z = y * targets + (1 - y) * (1 - targets)

    return -td.sumover(td.log(z))


def gradfun(w):

    return td.gradient(lossfun)(w)


def main():

    w = td.zeros(
           (td.IndexLit("j",3), ), 
           dtype="float64"
        )

    print("\n\nInitial loss: ", td.asdata(lossfun(w)))

    for i in range(100):
        w = w - gradfun(w) * 0.01

    print("Optimized loss: ", td.asdata(lossfun(w)))


main()


"""
cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

cProfile.run('main()', sort='tottime')

"""







