#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.array.core     as tcore
import tadpole.array.backends as tbackends




# --- Array data ------------------------------------------------------------ #

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




def array_dat(datafun):

    def wrap(backend, shape, **opts):

        data = datafun(shape, **opts)
        data = {
                "numpy": np.asarray,
                #"torch": torch.as_tensor, 
               }[backend](data)

        backend = tbackends.get(backend)
        array   = tcore.Array(backend, data)

        return ArrayData(array, backend, data, shape, opts)

    return wrap




# --- ArraySpace data ------------------------------------------------------- #

ArraySpaceData = collections.namedtuple("ArraySpaceData", [
                    "space", "backend", "dtype", "shape",
                 ])


def array_space_dat(backend, dtype, shape):

    space = tcore.ArraySpace(backend, dtype, shape)

    return ArraySpaceData(space, backend, dtype, shape)











































