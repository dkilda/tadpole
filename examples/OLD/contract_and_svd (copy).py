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

import timeit
import cProfile


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
    M[i] = M[i] / td.amax(M[i])**2

H = {}
for i in (1,2):
    H[i] = td.randn(
              (ind[f"h2"], ind[f"k{i}"], ind[f"b{i}"]), 
              dtype="complex128"
           )
    H[i] = H[i] / td.amax(H[i])**2



def _update(x, y):

    U, S, VH, _ = td.linalg.svd(x @ H[1], sind="s", rinds=(ind["m2"], ind["h2"]))  

    sind = IndexLit("s", S.size)
    R    = td.contract(S, VH, product=(sind, ind["m2"], ind["h2"]))
    yHR  = td.contract(R, y, H[2])

    indm2 = IndexGen("m2", len(sind))

    xout = x   # td.reindex(x,   {sind: indm2, ind["b1"]: ind["k1"]})
    yout = yHR # td.reindex(yHR, {sind: indm2, ind["b2"]: ind["k2"]})

    return xout @ yout

    

def update(x, y):

    z    = x @ y
    zout = _update(x, y)
        
    return (zout @ zout.H) / (z @ z.H) 



def gradX(fun):

    return lambda x, y: td.gradient(update, 0)(x, y) @ td.space(x).ones()



def gradY(fun):

    return lambda x, y: td.gradient(update, 1)(x, y) @ td.space(y).ones()



def comp(msg, expr):

    print(msg, td.asdata(expr))



def main():

    comp("updated tensors:    ", update(M[1], M[2]))
    comp("update gradient-x:  ", gradX(update)(M[1], M[2]))   
    comp("update gradient-y:  ", gradY(update)(M[1], M[2]))  
    comp("update gradient-xx: ", gradX(gradX(update))(M[1], M[2])) 
    comp("update gradient-yy: ", gradY(gradY(update))(M[1], M[2])) 
    comp("update gradient-xy: ", gradX(gradY(update))(M[1], M[2])) 
    comp("update gradient-yx: ", gradY(gradX(update))(M[1], M[2])) 



cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

#cProfile.run('main()', sort='tottime')




























