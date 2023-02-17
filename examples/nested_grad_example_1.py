#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import tadpole as tad




# val = 30.0, g = 6.0 

def fun(x, x0=3.0):
  def funY(y):
      return tad.mul(x, tad.mul(y, y)) 
  return tad.grad(funY)(x0)




x = 5.0

val = fun(x)
print(f"\nValue: {val}")

g = tad.grad(fun)(x)
print(f"\nGradient wrt 0: {g}")





















































































