#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from tadpole.autodiff.graph import differentiable, nondifferentiable


# --- Wrappers for differentiable functions --------------------------------- #

@differentiable
def sin(x):
    return np.sin(x)

@differentiable
def cos(x):
    return np.cos(x)

@differentiable
def neg(x):
    return -x

@differentiable
def add(x, y):
    return x + y

@differentiable
def sub(x, y):
    return x - y

@differentiable
def mul(x, y):
    return x * y




# --- Wrappers for non-differentiable functions ----------------------------- #

@nondifferentiable
def floor(x, n):
    return x // n

@nondifferentiable
def equals(x, y):
    return x == y 




# --- Wrapper for gradient addition ----------------------------------------- #

def add_grads(net_g, g): # TODO impl and use add() function, with @diffable decorator 
                         #      (or overload __add__ operator to make it @diffable)
    if net_g is None:  
       return g

    return add(net_g, g)













































