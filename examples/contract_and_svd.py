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


"""

Differentiation of a tensor network update involving contraction 
with a 2-site MPO followed by SVD. The loss function is defined as 
an overlap containing the updated tensors.

The example demonstrates the calculation of both the first- and 
the second-order gradients wrt different sites.

"""


d  = 2
D  = 10
DH = 5


ind = {}
def make_index(tags, size):
    ind[tags] = IndexGen(tags, size)


for i in "12":
    make_index(f"k{i}", d)
    make_index(f"b{i}", d)

for i in "123":
    make_index(f"m{i}", D)

make_index(f"h2", DH)


M = {}
for i in (1,2):
    M[i] = td.randn(
              (ind[f"m{i}"], ind[f"m{i+1}"], ind[f"k{i}"]), 
              dtype="complex128"
           )
    M[i] = M[i] / td.amax(M[i])


H = {}
for i in (1,2):
    H[i] = td.randn(
              (ind[f"h2"], ind[f"k{i}"], ind[f"b{i}"]), 
              dtype="complex128"
           )
    H[i] = H[i] / td.amax(H[i])



def update(x, y):

    U, S, VH, _ = td.linalg.svd(x @ H[1], sind="s", rinds=(ind["m2"], ind["h2"]))  

    sind = IndexLit("s", S.size)
    R    = td.contract(S, VH, product=(sind, ind["m2"], ind["h2"]))
    yHR  = td.contract(R, y, H[2])

    m2 = IndexGen("m2", len(sind))

    xout = td.reindex(U,   {sind: m2, ind["b1"]: ind["k1"]})
    yout = td.reindex(yHR, {sind: m2, ind["b2"]: ind["k2"]})

    return xout @ yout 
    


def overlap(x, y):

    z    = x @ y
    zout = update(x, y)
        
    return (zout @ z.H) / (z @ z.H) 



def gradX(fun):

    return lambda x, y: td.gradient(fun, 0)(x, y) @ td.space(x).ones()



def gradY(fun):

    return lambda x, y: td.gradient(fun, 1)(x, y) @ td.space(y).ones()



def comp(msg, expr):

    print(msg, td.asdata(expr))

  

def main():

    comp("overlap:             ", overlap(M[1], M[2]))
    comp("overlap gradient-x:  ", gradX(overlap)(M[1], M[2]))
    comp("overlap gradient-y:  ", gradY(overlap)(M[1], M[2]))   
    comp("overlap gradient-xx: ", gradX(gradX(overlap))(M[1], M[2]))
    comp("overlap gradient-yy: ", gradY(gradY(overlap))(M[1], M[2]))  
    comp("overlap gradient-xy: ", gradX(gradY(overlap))(M[1], M[2]))
    comp("overlap gradient-yx: ", gradY(gradX(overlap))(M[1], M[2]))  



main()



























