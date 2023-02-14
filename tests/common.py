#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
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

        yield common.amap(typ, xs)

        for xcombo in itertools.product(*xs):
            yield common.amap(typ, xcombo) 

    return wrap



















































