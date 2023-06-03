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


i = td.IndexGen("i",2)
j = td.IndexGen("j",2)
a = td.IndexGen("a",20)
b = td.IndexGen("b",20)
c = td.IndexGen("c",5)
d = td.IndexGen("d",5)
e = td.IndexGen("e",20)
f = td.IndexGen("f",20)

H = td.randn((c,d,i,j), dtype="complex128")
K = td.randn((a,b,i),   dtype="complex128")
B = K.C(e,f,j)
I = td.ones((e,f,j))

FL = td.randn((a,c,e), dtype="complex128")
FR = td.randn((b,d,f), dtype="complex128")


def con(x, y):
    return td.contract(x, H, y, FL, FR) / td.contract(x, x.C)


def gcon(x, y):
    return td.contract(td.gradient(con, 1)(x, y), I) 


def compute(msg, expr):
    print(msg, td.asdata(expr))


def main():
    compute("contraction:          ", con(K, B))
    compute("contraction gradient: ", gcon(K, B))    


cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

cProfile.run('main()', sort='tottime')




























