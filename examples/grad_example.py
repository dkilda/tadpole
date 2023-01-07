#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole as tad


def fun(x,y):
    return tad.sin(tad.cos(x)) * tad.cos(y) + tad.sin(x)


x = 2.7
y = 5.1

val = fun(x, y)
print("\nValue: {val}")

g = tad.grad(fun, 0)(x, y)
print("\nGradient wrt 0: {g}")

g = tad.grad(fun, 1)(x, y)
print("\nGradient wrt 1: {g}")














































































