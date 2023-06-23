#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy   as np
import tadpole as td

from tadpole import (
   IndexGen,
)


"""
Tensor contraction, example-1: 
a function involving tensor contraction and its gradients.

"""



def contract(x, y):
    z = td.contract(x, y)
    return td.contract(z.H, z)


def grad(fun, i=0):
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


x = x / td.amax(x)
y = y / td.amax(y)


def fun(x, y):

    def con(u, v):
        du = grad(contract)(u, v) 
        return contract(du, v) / contract(u, v)**2

    dx = grad(con, 0)(x, y) + grad(con, 1)(y, x)
    dy = grad(con, 1)(x, y) - grad(con, 0)(y, x)

    return contract(dx, dy)


def main():

    compute("Contraction C(x,y): ", contract(x, y))
    print("Evaluate contraction")
    print("C[u,v] = (d(u @ v)/du @ v) / (u @ v)^2, ") 
    print("F[x,y] = d(C[x,y] + C[y,x])/dx @ d(C[x,y] - C[y,x])/dy: ")
    compute(" ", fun(x,y))  


main()
























