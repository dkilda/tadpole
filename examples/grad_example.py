#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import tadpole as tad

def err(x,y):
    return abs(x-y) < 1e-10


def fun(x,y):
    return tad.add(tad.mul(tad.sin(tad.cos(x)), tad.cos(y)), tad.sin(x)) 


x = 2.7
y = 5.1

val = fun(x, y)
print(f"\nValue: {val}, {err(val, 0.13034543157)}")       # ans = 0.13034543157

g = tad.grad(fun, 0)(x, y) 
print(f"\nGradient wrt 0: {g}, {err(g, -1.00397095286)}") # ans = -1.00397095286

g = tad.grad(fun, 1)(x, y) 
print(f"\nGradient wrt 1: {g}, {err(g, -0.72755303459)}") # ans = -0.72755303459














































































