#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np




# --- Helpers --------------------------------------------------------------- #

def make_tuple(creator, n):

    return tuple(creator() for _ in range(n))




def map_tuple(creator, n):

    return tuple(map(creator, range(n)))




# --- Equality comparisons -------------------------------------------------- #

def value_eq(this, other, rtol=2**(-16), atol=2**(-32)):

    return np.allclose(this, other, rtol=rtol, atol=atol)



