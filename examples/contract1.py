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


def contract(x, y):
    z = td.contract(x, y)
    return td.contract(z.H, z)
 

def grad(fun, i):
    return td.gradient(fun, i)


def compute(msg, expr):
    print(msg, td.asdata(expr))
    



i = td.IndexGen("i",3)
j = td.IndexGen("j",4)
k = td.IndexGen("k",6)
l = td.IndexGen("l",2)
m = td.IndexGen("m",5)

x = td.randn((i,j,k))
y = td.randn((k,l,m))




def fun(x, y):

    def con(u, v):
        return contract(grad(contract, 0)(u, v), v) / contract(u, v)**2

    gx = grad(con, 0)(x, y) + grad(con, 1)(y, x)
    gy = grad(con, 1)(x, y) - grad(con, 0)(y, x)

    return contract(gx, gy)


def main():

    compute("\n\ncontraction C(x,y): ", contract(x, y))
    compute("\n\ngradient dC/dx:     ", grad(contract, 0)(x, y))
    compute("gradient dC/dy:         ", grad(contract, 1)(x, y))
    compute("\n\ngradient d2C/dx2:   ", grad(grad(contract, 0), 0)(x, y))
    compute("gradient d2C/dy2:       ", grad(grad(contract, 1), 1)(x, y))
    compute("\n\ngradient d2C/dxdy:  ", grad(grad(contract, 0), 1)(x, y))
    compute("gradient d2C/dydx:      ", grad(grad(contract, 1), 0)(x, y))
    print("\n\nEvaluate ")
    print("C[u,v] = (d(u @ v)/du @ v) / (u @ v)^2, ") 
    print("F[x,y] = d(C[x,y] + C[y,x])/dx @ d(C[x,y] - C[y,x])/dy: ")
    compute(" ", fun(x,y))  



cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

#cProfile.run('main()', sort='tottime')




























