#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy   as np
import tadpole as td

from tadpole import (
   IndexGen,
   IndexLit,
)

from util import (
   tprint
   TensorCollection,
   ArgsTensorCollection,
   ArgsTensor,
   Optimize,
)

import timeit
import cProfile


d = 10
D = 5
L = 4


ind = {}
def make_index(tags, size):
    ind[tags] = IndexGen(tags, size)


for n in "udlr":
    make_index(f"D{n}", D)


for l in range(1,L+1):
    make_index(f"d{l}", d)


M    = {}
M[1] = td.randn((ind["d1"], ind["Du"], ind["Dl"]))
M[2] = td.randn((ind["d2"], ind["Du"], ind["Dr"]))
M[3] = td.randn((ind["d3"], ind["Dd"], ind["Dl"]))
M[4] = td.randn((ind["d4"], ind["Dd"], ind["Dr"]))

M = TensorCollection(M)


target = td.randuniform((ind["d1"], ind["d2"], ind["d3"], ind["d4"]))
target = target / td.amax(target)




def distance(*ts):

    result = td.contract(*ts)

    return td.norm(result - target, linds=(ind["d1"], ind["d2"])) 




def main():

    tprint("initial distance: ", distance(*M.values()))

    optimize = Optimize(distance, ArgsTensorCollection(M))
    Mout     = optimize().astensorcollection()

    tprint("optimized distance: ", distance(*Mout.values()))




cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

# cProfile.run('main()', sort='tottime')



