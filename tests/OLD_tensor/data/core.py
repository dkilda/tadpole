#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tests.tensor.fakes as fake
import tests.tensor.data  as data

import tadpole.tensor.core     as core
import tadpole.tensor.backends as backends




# --- Basis data ------------------------------------------------------------ #

BasisData = collections.namedtuple("BasisData", [
                    "tensors", "datas", "idxs", 
                    "backend", "shape", "dtype",
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
    tensors = [core.Tensor(backend, x)          for x in xs]

    return BasisData(tensors, xs, idxs, backend, shape, dtype)




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

    xs      = [backend.asarray(x, dtype=dtype) for x in xs]
    tensors = [core.Tensor(backend, x)         for x in xs]

    return BasisData(tensors, xs, idxs, backend, shape, dtype)




# --- Sample data ----------------------------------------------------------- #

SampleData = collections.namedtuple("SampleData", [
               "tensor", "data", "backend", "shape", "dtype", "opts"
            ])




def zeros_dat_001(backend, dtype="complex128"):

    backend = backends.get(backend)
    shape   = (2,3)

    data = np.array([[0,0,0],[0,0,0]])
    data = backend.asarray(data, dtype=dtype)

    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, backend, shape, dtype, {})




def ones_dat_001(backend, dtype="complex128"):

    backend = backends.get(backend)
    shape   = (2,3)

    data = np.array([[1,1,1],[1,1,1]])
    data = backend.asarray(data, dtype=dtype)

    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, backend, shape, dtype, {})




def rand_real_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "float64"

    np.random.seed(seed)

    data  = np.random.rand(*shape).astype(dtype)
    data  = backend.asarray(data, dtype=dtype)
    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, backend, shape, dtype, {})




def rand_complex_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "complex128"

    np.random.seed(seed)

    data  =      np.random.rand(*shape).astype(dtype) 
    data += 1j * np.random.rand(*shape).astype(dtype)

    data  = backend.asarray(data, dtype=dtype)
    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, backend, shape, dtype, {})




def randn_real_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "float64"

    np.random.seed(seed)

    data  = np.random.randn(*shape).astype(dtype)
    data  = backend.asarray(data, dtype=dtype)
    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, backend, shape, dtype, {})




def randn_complex_dat_001(backend, seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "complex128"

    np.random.seed(seed)

    data  =      np.random.randn(*shape).astype(dtype) 
    data += 1j * np.random.randn(*shape).astype(dtype)

    data  = backend.asarray(data, dtype=dtype)
    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, backend, shape, dtype, {})




def randuniform_real_dat_001(backend, boundaries=(0,1), seed=1):

    backend = backends.get(backend)
    shape   = (2,3)
    dtype   = "float64"

    np.random.seed(seed)

    args = (boundaries[0], boundaries[1], tuple(shape))

    data  = np.random.uniform(*args).astype(dtype)
    data  = backend.asarray(data, dtype=dtype)
    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, 
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
    tensor = core.Tensor(backend, data)

    return SampleData(tensor, data, 
                      backend, shape, dtype, 
                      {"boundaries": boundaries})




# --- Tensor data ----------------------------------------------------------- #

TensorData = collections.namedtuple("TensorData", [
                "tensor", "backend", "data", 
                "shape", "opts",
             ])




def randn(shape, dtype="complex128", seed=1):

    np.random.seed(seed)

    if 'complex' in dtype:
       return np.random.randn(*shape).astype(dtype) \
              + 1j * np.random.randn(*shape).astype(dtype)

    return np.random.randn(*shape).astype(dtype)




def randn_pos(shape, nvals=1, defaultval=0, dtype="complex128", seed=1):

    np.random.seed(seed)

    size      = np.prod(shape)
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




def tensor_dat(datafun):

    def wrap(backend, shape, **opts):

        data = datafun(shape, **opts)
        data = {
                "numpy": np.asarray,
                #"torch": torch.as_tensor, 
               }[backend](data)

        backend = backends.get(backend)
        tensor  = core.Tensor(backend, data)

        return TensorData(tensor, backend, data, shape, opts)

    return wrap




# --- TensorSpace data ------------------------------------------------------ #

TensorSpaceData = collections.namedtuple("TensorSpaceData", [
                     "space", "backend", "shape", "dtype",
                  ])


def tensor_space_dat(backend, shape, dtype):

    space = core.TensorSpace(backend, shape, dtype)

    return TensorSpaceData(space, backend, shape, dtype)



