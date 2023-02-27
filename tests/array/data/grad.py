#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.array.core     as tcore
import tadpole.array.grad     as tgrad
import tadpole.array.backends as tbackends




# --- Sparse gradient data -------------------------------------------------- #

SparseGradData = collections.namedtuple("SparseGradData", [
                    "grad", 
                    "space", "idxs", "vals", 
                    "backend", "dtype", "shape",
                 ])




def sparse_grad_dat(idxs, vals, backend, dtype, shape):

    space = tcore.ArraySpace(backend, dtype, shape)
    grad  = tgrad.SparseGrad(space, idxs, vals)

    return SparseGradData(
                          grad, 
                          space, idxs, vals,
                          backend, dtype, shape,
                         )
 


