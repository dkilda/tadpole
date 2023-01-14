#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import tadpole as tad


def fun(x,y):
    return tad.add(tad.mul(tad.sin(tad.cos(x)), tad.cos(y)), tad.sin(x)) # tad.add(x, y) # tad.add(tad.mul(tad.sin(tad.cos(x)), tad.cos(y)), tad.sin(x))


x = 2.7
y = 5.1

val = fun(x, y)
print(f"\nValue: {val}")   # ans = 0.13034543157

g = tad.grad(fun, 0)(x, y) # ans = -1.00397095286
print(f"\nGradient wrt 0: {g}")

g = tad.grad(fun, 1)(x, y) # ans = -0.72755303459
print(f"\nGradient wrt 1: {g}")














































































