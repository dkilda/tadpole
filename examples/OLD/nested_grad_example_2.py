#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import tadpole as tad



# val = 5.0, g = 1.0

def fun(x):
   def funY(y):
       return tad.mul(x, y)
   return tad.grad(funY)(x)


x = 5.0

val = fun(x)
print(f"\nValue: {val}")

g = tad.grad(fun)(x)
print(f"\nGradient wrt 0: {g}")





















































































