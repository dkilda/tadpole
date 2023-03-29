#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.tensor.fakes as fake
import tests.tensor.data  as data

import tadpole.tensor.core     as core
import tadpole.tensor.grad     as grad
import tadpole.tensor.backends as backends




# --- Sparse gradient data -------------------------------------------------- #

SparseGradData = collections.namedtuple("SparseGradData", [
                    "grad",     "dense",  "densedata",
                    "space",    "idxs",   "vals", 
                    "backend",  "dtype",  "shape",
                 ])




def sparse_grad_dat(backend, shape, dtype, idxs, vals):

    space = core.TensorSpace(backend, shape, dtype)
    g     = space.sparse(idxs, vals)

    def densefun(shape, dtype): 
        data       = np.zeros(shape, dtype=dtype)
        data[idxs] = vals
        return data

    x = data.tensor_dat(densefun)(backend, shape, dtype=dtype)

    return SparseGradData(
                          g,         x.tensor, x.data,
                          space,     idxs,     vals,
                          x.backend, dtype,    shape,
                         )
 



def sparse_grad_dat_001(backend):

    dtype = "float64"
    shape = (2,3,4) 

    idxs = (
            ((1,0,1),), 
            ((0,2,0),),
            ((2,1,3),),
           )

    np.random.seed(1)
    vals = np.random.randn(len(idxs))

    dense        = np.zeros(shape, dtype=dtype)
    dense[1,0,2] = vals[0]
    dense[0,2,1] = vals[1]
    dense[1,0,3] = vals[2]

    space = core.TensorSpace(backend, shape, dtype)
    g     = space.sparse(idxs, vals)

    def densefun(shape, dtype): 
        return dense

    x = data.tensor_dat(densefun)(backend, shape, dtype=dtype)

    return SparseGradData(
                          g,         x.tensor, x.data,
                          space,     idxs,     vals,
                          x.backend, dtype,    shape,
                         )
 



def sparse_grad_dat_002(backend):

    dtype = "complex128"
    shape = (2,3,4) 

    idxs = (
            ((1,0,1),), 
            ((0,2,0),),
            ((2,1,3),),
           )

    np.random.seed(1)
    vals = (np.random.randn(len(idxs)) 
            + 1j * np.random.randn(len(idxs)))

    dense        = np.zeros(shape, dtype=dtype)
    dense[1,0,2] = vals[0]
    dense[0,2,1] = vals[1]
    dense[1,0,3] = vals[2]

    space = core.TensorSpace(backend, shape, dtype)
    g     = space.sparse(idxs, vals)

    def densefun(shape, dtype): 
        return dense

    x = data.tensor_dat(densefun)(backend, shape, dtype=dtype)

    return SparseGradData(
                          g,         x.tensor, x.data,
                          space,     idxs,     vals,
                          x.backend, dtype,    shape,
                         )




