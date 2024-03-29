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
Tensor contraction, example-2: 
a function involving tensor contraction and its gradients.

"""


def grad(fun, i=0):
    return td.gradient(fun, i)


def compute(msg, expr):
    print(msg, td.asdata(expr))
    

def fun(x, y, **opts):

    def dot(x, y):
        z = td.contract(x, y, **opts)
        return td.contract(z.H, z) 

    def con(u, v):
        du = grad(dot)(u, v)
        return dot(du, v) / dot(u, v)

    gx = grad(con, 0)(x, y) + grad(con, 1)(y, x)
    gy = grad(con, 1)(x, y) - grad(con, 0)(y, x)

    return dot(gx, gy) / dot(x, y) **2


I = td.IndexGen("I",3)
J = td.IndexGen("J",4)
L = td.IndexGen("L",6)
M = td.IndexGen("M",2)
N = td.IndexGen("N",5)
O = td.IndexGen("O",4)

i = td.IndexGen("i",10)
j = td.IndexGen("j",10)
l = td.IndexGen("l",10)
m = td.IndexGen("m",10)
n = td.IndexGen("n",10)
o = td.IndexGen("o",10)

x = td.randn((I,J,L,M,i,j,l,m))
y = td.randn((N,O,J,L,n,o,j,l))

x = x / td.amax(x)
y = y / td.amax(y)


def main():

    print("Evaluate contraction")
    print("C[u,v] = (d(u @ v)/du @ v) / (u @ v), ") 
    print("F[x,y] = d(C[x,y] + C[y,x])/dx @ d(C[x,y] - C[y,x])/dy / (x @ y)^2: ")
    compute(" ", fun(x, y, product=(I,N,O,M,i,n,o,m)))  


main()



























