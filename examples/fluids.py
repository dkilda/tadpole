#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import scipy.optimize
import scipy.misc

import matplotlib
import matplotlib.pyplot as plt
import os

import numpy   as np
import tadpole as td

from tadpole import (
   IndexGen,
   IndexLit,
)



'''
# def project(vx, vy):
# ix, iy = np.meshgrid(np.arange(3), np.arange(4))

    
x   = td.randn((IndexLit("i",3), IndexLit("i",4)))
idx = (ix, iy) #(1, slice(2))


def fun(x, idx):
    return x[idx]


print("Value: ", td.asdata(fun(x, idx)))
print("Grad:  ", td.asdata(td.gradient(fun)(x, idx)))

'''







































