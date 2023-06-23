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
   tprint,
   TensorCollection,
   ArgsTensorCollection,
   ArgsTensor,
   Optimize,
)


"""

Tensor fitting: we demonstrate the fitting of a 4-tensor ring network
(a decomposition where each tensor has local dimension d) to a single
target tensor (a dense representation with (d, d, d, d) dimensions) 
by optimizing the distance between them. 

This example utilizes L-BFGS-B optimizer, where gradients are computed 
using the tadpole autodiff.

"""




d = 3 
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
M[1] = td.randn((ind["d1"], ind["Du"], ind["Dl"]), dtype="float64")
M[2] = td.randn((ind["d2"], ind["Du"], ind["Dr"]), dtype="float64")
M[3] = td.randn((ind["d3"], ind["Dd"], ind["Dl"]), dtype="float64")
M[4] = td.randn((ind["d4"], ind["Dd"], ind["Dr"]), dtype="float64")

M = TensorCollection(M)


target = td.randuniform(
            (ind["d1"], ind["d2"], ind["d3"], ind["d4"]), 
            (0,1), 
            dtype="float64"
         )
target = target / td.amax(target)




def distance(*ts):

    result = td.contract(*ts)

    return td.linalg.norm(result - target, linds=(ind["d1"], ind["d2"])) 




def main():

    tprint("initial distance: ", distance(*M))

    optimize = Optimize(distance, ArgsTensorCollection(M))
    Mout     = optimize(
                  options={"maxiter": 100}
               ).astensorcollection()

    tprint("optimized distance: ", distance(*Mout))



main()


