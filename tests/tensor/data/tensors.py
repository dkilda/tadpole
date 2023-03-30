#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.array.backends as backends

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tadpole.tensor.types import (
   Pluggable,
   Tensor, 
   Space,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




# --- Indices data ---------------------------------------------------------- #

IndicesData = collections.namedtuple("TensorSpaceData", [
                 "inds", "indlist", "names", "shape", 
              ])




def indices_dat(names, shape):

    if names is None:
       names = "".join(tuple(zip(*zip(shape, "ijklmnpqr")))[1])

    indlist = [IndexGen(name, size) for name, size in zip(names, shape)]
    inds    = Indices(*indlist)

    return IndicesData(inds, indlist, names, shape)




# --- Basis data ------------------------------------------------------------ #

BasisData = collections.namedtuple("BasisData", [
               "tensors", "arrays", "datas", "pos",
               "inds", "shape", "backend", "dtype"
            ])




def basis_dat(datafun):

    def wrap(backend, *args, **opts):
              
        w = datafun(backend, *args, **opts)
        v = indices_dat(None, w.shape)

        tensors = [tn.TensorGen(array, v.inds) for array in w.arrays] 

        return SampleData(
           tensors, w.arrays, w.datas, w.idxs,
           v.inds, w.shape, w.backend, w.dtype, w.opts
        )

    return wrap




# --- Sample data ----------------------------------------------------------- #

SampleData = collections.namedtuple("SampleData", [
                "tensor", "array", "data", 
                "inds", "shape", "backend", "dtype", "opts"
             ])




def sample_dat(datafun):

    def wrap(backend, *args, **opts):
              
        w = datafun(backend, *args, **opts)
        v = indices_dat(None, w.shape)

        tensor = tn.TensorGen(w.array, v.inds)

        return SampleData(
           tensor, w.array, w.data, 
           v.inds, w.shape, w.backend, w.dtype, w.opts
        )

    return wrap




# --- Gradient data --------------------------------------------------------- #

SparseGradData = collections.namedtuple("SparseGradData", [
                    "grad", 
                    "tensor", "array", "data",
                    "space",  "pos",   "vals", 
                    "inds",   "shape", "dtype", "backend",  "opts",
                 ])




def sparse_grad_dat(backend, indnames, shape, dtype, pos, vals, seed=1):

    def densefun(shape, dtype):

        backend = backends.get(backend) 
        data    = backend.zeros(shape, dtype=dtype)

        for p, v in zip(pos, vals):
            data[p] = v

        return data

    x = tensor_dat(densefun)(
           backend, indnames, shape, dtype=dtype, seed=seed
        )
    w = data.arrayspace_dat(backend, shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, pos, vals)

    return SparseGradData(
                          grad,         
                          x.tensor, x.array,  x.data,
                          space,    pos,      vals,
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




def sparse_grad_dat_001(backend, dtype, seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)

    pos = (
           ((1,0,1),), 
           ((0,2,0),),
           ((2,1,3),),
          )
    vals = backend.randn((len(pos),), dtype=dtype, seed=seed)  

    dense        = backend.zeros(shape, dtype=dtype)
    dense[1,0,2] = vals[0]
    dense[0,2,1] = vals[1]
    dense[1,0,3] = vals[2]

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
        )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, pos, vals)

    return SparseGradData(
                          grad,         
                          x.tensor, x.array,  x.data,
                          space,    pos,      vals,
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




# --- Tensor data ----------------------------------------------------------- #

TensorData = collections.namedtuple("TensorData", [
                "tensor", "array", "data", 
                "inds", "shape", "backend", "opts",
             ])




def tensor_dat(datafun):

    def wrap(backend, indnames, shape, **opts):

        v = indices_dat(indnames, shape)
        w = data.array_dat(datafun)(backend, shape, **opts)

        tensor = tn.TensorGen(w.array, v.inds)

        return TensorData(
           tensor, w.array, w.data, 
           v.inds, w.shape, w.backend, w.opts
        )

    return wrap




# --- TensorSpace data ------------------------------------------------------ #

TensorSpaceData = collections.namedtuple("TensorSpaceData", [
                     "tensorspace", "arrayspace",
                     "inds", "shape", "dtype", "backend",
                  ])




def tensorspace_dat(backend, indnames, shape, dtype):

    v = indices_dat(indnames, shape)
    w = data.arrayspace_dat(backend, shape, dtype)

    space = sp.TensorSpace(w.arrayspace, w.inds)

    return TensorSpaceData(
       space, w.space, v.inds, 
       w.shape, w.dtype, w.backend
    ) 




