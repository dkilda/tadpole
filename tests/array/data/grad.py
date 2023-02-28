#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.array.core     as core
import tadpole.array.grad     as grad
import tadpole.array.backends as backends




# --- Sparse gradient data -------------------------------------------------- #

SparseGradData = collections.namedtuple("SparseGradData", [
                    "grad", "dense",
                    "space", "idxs", "vals", 
                    "backend", "dtype", "shape",
                 ])




def sparse_grad_dat(backend, shape, dtype, idxs, vals):

    space = core.ArraySpace(backend, shape, dtype)
    g     = grad.SparseGrad(space, idxs, vals)

    def densefun(shape, dtype):
 
        data = np.zeros(shape, dtype=dtype)

        for idx, val in zip(idxs, vals):
            data[idx] = val

        return data

    x = data.array_dat(densefun)(backend, shape, dtype=dtype)

    return SparseGradData(
                          g, x.array,
                          space, idxs, vals,
                          x.backend, dtype, shape,
                         )
 


