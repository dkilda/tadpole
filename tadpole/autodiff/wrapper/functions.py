#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tadpole.autodiff.graph as tdgraph 


# --- Wrappers for differentiable functions --------------------------------- #

@tdgraph.differentiable
def sin(x):
    return np.sin(x)

@tdgraph.differentiable
def cos(x):
    return np.cos(x)

@tdgraph.differentiable
def neg(x):
    return -x

@tdgraph.differentiable
def add(x, y):
    return x + y

@tdgraph.differentiable
def sub(x, y):
    return x - y

@tdgraph.differentiable
def mul(x, y):
    return x * y




# --- Wrappers for non-differentiable functions ----------------------------- #

@tdgraph.nondifferentiable
def floor(x, n):
    return x // n

@tdgraph.nondifferentiable
def equals(x, y):
    return x == y 













































