#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import tadpole as tad


def fun(x):
   def funY(y):
       return tad.mul(x, y)
   return tad.grad(funY)(x)




x = 2.7
y = 5.1

val = fun(x, y)
print(f"\nValue: {val}")

g = tad.grad(fun, 0)(x, y)
print(f"\nGradient wrt 0: {g}")

g = tad.grad(fun, 1)(x, y)
print(f"\nGradient wrt 1: {g}")



















































































