#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np




# --- Comparison of approximate data ---------------------------------------- #

def allclose(x, y, rtol=None, atol=None, **opts):

    if rtol is None: rtol = 1e-5 
    if atol is None: atol = 1e-8 

    return np.allclose(x, y, rtol=rtol, atol=atol, **opts)




# --- Comparison of iterables of approximate data --------------------------- #

def allallclose(xs, ys, **opts):

    return all(allclose(x, y, **opts) for x, y in zip(xs, ys))
































