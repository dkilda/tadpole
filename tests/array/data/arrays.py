#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.array.unary    as unary
import tadpole.array.binary   as binary
import tadpole.array.nary     as nary
import tadpole.array.void     as void
import tadpole.array.space    as sp
import tadpole.array.backends as backends




# --- Basis data ------------------------------------------------------------ #

BasisData = collections.namedtuple("BasisData", [
               "arrays", "datas", "idxs", "backend", "shape", "dtype",
            ])




def basis_real_dat_001(backend):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "float64"

    idxs = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]

    xs = [
          np.array([[1,0,0],[0,0,0]]),
          np.array([[0,1,0],[0,0,0]]),
          np.array([[0,0,1],[0,0,0]]),
          np.array([[0,0,0],[1,0,0]]),
          np.array([[0,0,0],[0,1,0]]),
          np.array([[0,0,0],[0,0,1]]),
         ]

    xs     = [backend.asarray(x, dtype=dtype) for x in xs]
    arrays = [unary.Array(backend, x)         for x in xs]

    return BasisData(arrays, xs, idxs, backend, shape, dtype)




def basis_complex_dat_001(backend):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "complex128"

    idxs = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]

    xs = [
          np.array([[1,0,0],[0,0,0]]),
          np.array([[1j,0,0],[0,0,0]]),
          np.array([[0,1,0],[0,0,0]]),
          np.array([[0,1j,0],[0,0,0]]),
          np.array([[0,0,1],[0,0,0]]),
          np.array([[0,0,1j],[0,0,0]]),
          np.array([[0,0,0],[1,0,0]]),
          np.array([[0,0,0],[1j,0,0]]),
          np.array([[0,0,0],[0,1,0]]),
          np.array([[0,0,0],[0,1j,0]]),
          np.array([[0,0,0],[0,0,1]]),
          np.array([[0,0,0],[0,0,1j]]),
         ]

    xs     = [backend.asarray(x, dtype=dtype) for x in xs]
    arrays = [unary.Array(backend, x)         for x in xs]

    return BasisData(arrays, xs, idxs, backend, shape, dtype)




# --- Sample data ----------------------------------------------------------- #

SampleData = collections.namedtuple("SampleData", [
                "array", "data", "backend", "shape", "dtype", "opts"
             ])




def zeros_dat_001(backend, dtype="complex128", **opts):

    backend = backends.get(backend)
    shape   = (2,3)

    data = np.array([[0,0,0],[0,0,0]])
    data = backend.asarray(data, dtype=dtype)

    array = unary.Array(backend, data)

    return SampleData(array, data, backend, shape, dtype, {})




def ones_dat_001(backend, dtype="complex128"):

    backend = backends.get(backend)
    shape   = (2,3)

    data = np.array([[1,1,1],[1,1,1]])
    data = backend.asarray(data, dtype=dtype)

    array = unary.Array(backend, data)

    return SampleData(array, data, backend, shape, dtype, {})




def rand_real_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "float64"

    np.random.seed(seed)

    data  = np.random.rand(*shape).astype(dtype)
    data  = backend.asarray(data, dtype=dtype)
    array = unary.Array(backend, data)

    return SampleData(array, data, backend, shape, dtype, {})




def rand_complex_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "complex128"

    np.random.seed(seed)

    data  =      np.random.rand(*shape).astype(dtype) 
    data += 1j * np.random.rand(*shape).astype(dtype)

    data  = backend.asarray(data, dtype=dtype)
    array = unary.Array(backend, data)

    return SampleData(array, data, backend, shape, dtype, {})




def randn_real_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "float64"

    np.random.seed(seed)

    data  = np.random.randn(*shape).astype(dtype)
    data  = backend.asarray(data, dtype=dtype)
    array = unary.Array(backend, data)

    return SampleData(array, data, backend, shape, dtype, {})




def randn_complex_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "complex128"

    np.random.seed(seed)

    data  =      np.random.randn(*shape).astype(dtype) 
    data += 1j * np.random.randn(*shape).astype(dtype)

    data  = backend.asarray(data, dtype=dtype)
    array = unary.Array(backend, data)

    return SampleData(array, data, backend, shape, dtype, {})




def randuniform_int_dat_001(backend, boundaries=(0,1), seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "int"

    np.random.seed(seed)

    args = (boundaries[0], boundaries[1], tuple(shape))

    data  = np.random.uniform(*args).astype(dtype)
    data  = backend.asarray(data, dtype=dtype)
    array = unary.Array(backend, data)

    return SampleData(array, data, 
                      backend, shape, dtype, 
                      {"boundaries": boundaries})



def randuniform_real_dat_001(backend, boundaries=(0,1), seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "float64"

    np.random.seed(seed)

    args = (boundaries[0], boundaries[1], tuple(shape))

    data  = np.random.uniform(*args).astype(dtype)
    data  = backend.asarray(data, dtype=dtype)
    array = unary.Array(backend, data)

    return SampleData(array, data, 
                      backend, shape, dtype, 
                      {"boundaries": boundaries})




def randuniform_complex_dat_001(backend, boundaries=(0,1), seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "complex128"

    np.random.seed(seed)

    args = (boundaries[0], boundaries[1], tuple(shape))

    data =       np.random.uniform(*args).astype(dtype) 
    data += 1j * np.random.uniform(*args).astype(dtype)

    data  = backend.asarray(data, dtype=dtype)
    array = unary.Array(backend, data)

    return SampleData(array, data, 
                      backend, shape, dtype, 
                      {"boundaries": boundaries})




# --- Array data ----------------------------------------------------------- #

ArrayData = collections.namedtuple("ArrayData", [
               "array", "backend", "data", "shape", "opts",
            ])




def randn(shape, dtype="complex128", seed=1):

    np.random.seed(seed)

    if 'complex' in dtype:
       return np.random.randn(*shape).astype(dtype) \
              + 1j * np.random.randn(*shape).astype(dtype)

    return np.random.randn(*shape).astype(dtype)




def randn_pos(shape, nvals=1, defaultval=0, dtype="complex128", seed=1):

    np.random.seed(seed)

    size      = int(np.prod(shape))
    positions = np.random.choice(np.arange(size), nvals, replace=False)

    if   defaultval == 0:
         data = np.zeros(size)
    else:
         data = defaultval * np.ones(size)

    if  'complex' in dtype:
         data[positions] = np.random.normal(size=nvals) \
                           + 1j * np.random.normal(size=nvals)
    else: 
         data[positions] = np.random.normal(size=nvals)

    return np.reshape(data, shape).astype(dtype)




def array_dat(datafun):

    def wrap(backend, shape, **opts):

        if  len(shape) == 0:

            data = datafun((1,), **opts)
            data = {
                    "numpy": np.asarray,
                    #"torch": torch.as_array, 
                   }[backend](data)
            data = {
                    "numpy": np.squeeze,
                    #"torch": torch.squeeze, 
                   }[backend](data)

        else:
            data = datafun(shape, **opts)
            data = {
                    "numpy": np.asarray,
                    #"torch": torch.as_array, 
                   }[backend](data)

        backend = backends.get(backend)
        array   = unary.Array(backend, data) 

        return ArrayData(array, backend, data, shape, opts)

    return wrap




# --- NArray data ----------------------------------------------------------- #

NArrayData = collections.namedtuple("NArrayData", [
                 "narray", "arrays", 
                 "backend", "datas", "shapes", "dtypes", "opts",
             ])




def narray_dat(datafun):

    def wrap(backend, shapes=None, dtypes=None, **opts):

        seed = opts.pop("seed", 1)

        if shapes is None:
           shapes = []

        if dtypes is None:
           dtypes = ["complex128"] * len(shapes) 

        ws = []
        for i in range(len(shapes)):

            w = array_dat(datafun)(
                      backend, shapes[i], dtype=dtypes[i], seed=seed+i, **opts
                   )
            ws.append(w)

        arrays  = [w.array for w in ws]
        datas   = [w.data  for w in ws]
        backend = backends.get(backend)

        if   len(shapes) == 0:
             narray = void.Array(backend)
        elif len(shapes) == 1:
             narray = unary.Array(backend, *datas)
        elif len(shapes) == 2:
             narray = binary.Array(backend, *datas) 
        else:
             narray = nary.Array(backend, *datas) 

        return NArrayData(
                  narray, arrays, 
                  backend, datas, shapes, dtypes, opts
               )

    return wrap




# --- ArraySpace data ------------------------------------------------------ #

ArraySpaceData = collections.namedtuple("ArraySpaceData", [
                    "space", "backend", "shape", "dtype",
                 ])




def arrayspace_dat(backend, shape, dtype):

    backend = backends.get(backend)
    space   = sp.ArraySpace(backend, shape, dtype)

    return ArraySpaceData(space, backend, shape, dtype)




