#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools

import numpy as np




# --- Equality comparisons -------------------------------------------------- #

def allclose(x, y, rtol=2**(-16), atol=2**(-32)):

    return np.allclose(x, y, rtol=rtol, atol=atol)




# --- Tuple-of-objects creation --------------------------------------------- #

def arepeat(creator, n): 

    return tuple(creator() for _ in range(n))


def arange(creator, n):

    return tuple(map(creator, range(n)))


def amap(creator, xs): 

    return tuple(map(creator, xs))




# --- Create objects from combinations of ctor args ------------------------- #   

def combos(typ):

    def wrap(*xs):

        repeated = set()

        for xcombo in itertools.product(*zip(*xs)):

            if xcombo in repeated:
               continue

            repeated.add(xcombo)
             
            yield typ(*xcombo)
  

    return wrap




# --- Create options dict from input kwargs (ignoring any None values) ------ #  

def options(**kwargs):

    return {k: v for k,v in kwargs.items() if v is not None}
        



