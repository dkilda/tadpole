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

IndicesData = collections.namedtuple("IndicesData", [
                 "inds", "indlist", "names", "shape", 
              ])




def indices_dat(names, shape):

    if names is None:
       names = "".join(tuple(zip(*zip(shape, "ijklmnpqr")))[1])

    indlist = [IndexGen(name, size) for name, size in zip(names, shape)]
    inds    = Indices(*indlist)

    return IndicesData(inds, indlist, names, shape)




def nindices_dat(names, shapes):

    names, sizes = util.concat(names), util.concat(shapes)
    names, sizes = zip(*tuple(set(zip(names, sizes))))

    return indices_dat(tuple(names), tuple(sizes))




# --- Basis data ------------------------------------------------------------ #

TensorBasisData = collections.namedtuple("TensorBasisData", [
                     "tensors", "arrays", "datas", "pos",
                     "inds", "shape", "backend", "dtype"
                  ])




def tensor_basis_dat(datafun):

    def wrap(backend, *args, **opts):
              
        w = datafun(backend, *args, **opts)
        v = indices_dat(None, w.shape)

        tensors = [tn.TensorGen(array, v.inds) for array in w.arrays] 

        return TensorBasisData(
           tensors, w.arrays, w.datas, w.idxs,
           v.inds, w.shape, w.backend, w.dtype
        )

    return wrap




# --- Sample data ----------------------------------------------------------- #

TensorSampleData = collections.namedtuple("TensorSampleData", [
                      "tensor", "array", "data", 
                      "inds", "shape", "backend", "dtype", "opts"
                   ])




def tensor_sample_dat(datafun):

    def wrap(backend, *args, **opts):
              
        w = datafun(backend, *args, **opts)
        v = indices_dat(None, w.shape)

        tensor = tn.TensorGen(w.array, v.inds)

        return TensorSampleData(
           tensor, w.array, w.data, 
           v.inds, w.shape, w.backend, w.dtype, w.opts
        )

    return wrap




# --- Null Gradient data -------------------------------------------------- #

NullGradData = collections.namedtuple("NullGradData", [
                  "grad",   "space", 
                  "tensor", "array", "data",  
                  "inds",   "shape", "dtype", "backend",  
                  "opts",
               ])




def nullgrad_dat(backend, indnames, shape, dtype="complex128"):

    w    = tensorspace_dat(backend, indnames, shape, dtype)
    grad = tn.NullGrad(w.tensorspace)

    data   = w.backend.zeros(shape, dtype=dtype)
    array  = ar.ArrayUnary(w.backend, data)    
    tensor = tn.TensorGen(array, w.inds)

    return NullGradData(
              grad,   w.tensorspace, 
              tensor, array,   data,
              w.inds, w.shape, w.dtype, w.backend, {}
           )




def nullgrad_dat_001(backend):

    return nullgrad_dat(backend, "ijk", (2,3,4), "complex128")




# --- Sparse Gradient data -------------------------------------------------- #

SparseGradData = collections.namedtuple("SparseGradData", [
                    "grad",   "space",
                    "tensor", "array", "data",
                    "vals",   "xvals", "valinds",  
                    "pos",    "xpos",  
                    "inds",   "shape", "dtype", "backend",  "opts",
                 ])




def sparsegrad_dat(backend, dtype="complex128", seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)
    valinds  = (IndexGen("Q",3),)

    pos  = [(1,0,2), (0,2,1), (1,0,3)]
    xpos = (
            ar.asarray([[[1,0,1]]], backend=backend), 
            ar.asarray([[[0,2,0]]], backend=backend),
            ar.asarray([[[2,1,3]]], backend=backend),
           )

    vals  = backend.randn((len(pos),), dtype=dtype, seed=seed)  
    xvals = ar.asarray(vals)

    dense = backend.zeros(shape, dtype=dtype)
    for i in range(len(pos)):
        dense[pos[i]] = vals[i]

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
    )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, xpos, xvals)

    return SparseGradData(
                          grad,     space,         
                          x.tensor, x.array,  x.data, 
                          vals,     xvals,    valinds,        
                          pos,      xpos,            
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




def sparsegrad_dat_001(backend, dtype="complex128", seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)
    valinds  = tuple()

    pos  = (1,0,2)
    xpos = (
            ar.asarray([[[1]]], backend=backend), 
            ar.asarray([[[0]]], backend=backend),
            ar.asarray([[[2]]], backend=backend),
           )

    vals  = backend.randn((1,), dtype=dtype, seed=seed)[0]  
    xvals = ar.asarray(vals)

    dense      = backend.zeros(shape, dtype=dtype)
    dense[pos] = vals

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
    )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, xpos, xvals)

    return SparseGradData(
                          grad,     space,         
                          x.tensor, x.array,  x.data, 
                          vals,     xvals,    valinds,           
                          pos,      xpos,        
                          x.inds,   x.shape,  dtype,  x.backend,  {},
                         )




def sparsegrad_dat_002(backend, dtype="complex128", seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)
    valinds  = tuple()

    vals  = backend.randn((1,), dtype=dtype, seed=seed)[0]
    xvals = ar.asarray(vals)

    pos  = (0,2,1)
    xpos = (
            ar.asarray([[[0]]], backend=backend), 
            ar.asarray([[[2]]], backend=backend),
            ar.asarray([[[1]]], backend=backend),
           )

    dense      = backend.zeros(shape, dtype=dtype)
    dense[pos] = vals

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
    )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, xpos, xvals)

    return SparseGradData(
                          grad,     space,         
                          x.tensor, x.array,  x.data, 
                          vals,     xvals,    valinds,            
                          pos,      xpos,        
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




def sparsegrad_dat_003(backend, dtype="complex128", seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)
    valinds  = tuple()

    vals  = backend.randn((1,), dtype=dtype, seed=seed)[0]
    xvals = ar.asarray(vals)  

    pos  = (1,0,3)
    xpos = (
            ar.asarray([[[1]]], backend=backend), 
            ar.asarray([[[0]]], backend=backend),
            ar.asarray([[[3]]], backend=backend),
           )

    dense      = backend.zeros(shape, dtype=dtype)
    dense[pos] = vals

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
    )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, xpos, xvals)

    return SparseGradData(
                          grad,     space,         
                          x.tensor, x.array,  x.data, 
                          vals,     xvals,    valinds,           
                          pos,      xpos,        
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




def sparsegrad_dat_004(backend, dtype="complex128", seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)

    vals  = backend.randn((1,2,2), dtype=dtype, seed=seed)  
    xvals = ar.asarray(vals)

    pos  = (slice(0,1), slice(1,3), slice(2,4))
    xpos = (
            ar.asarray([[[0,0], [0,0]]], backend=backend), 
            ar.asarray([[[1,1], [2,2]]], backend=backend),
            ar.asarray([[[2,3], [2,3]]], backend=backend),
           )

    dense      = backend.zeros(shape, dtype=dtype)
    dense[pos] = vals

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
    )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, xpos, xvals)

    valinds = (
               x.inds[0].resized(0,1), 
               x.inds[1].resized(1,3), 
               x.inds[2].resized(2,4),
              ) 

    return SparseGradData(
                          grad,     space,         
                          x.tensor, x.array,  x.data, 
                          vals,     xvals,    valinds,            
                          pos,      xpos,             
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




def sparsegrad_dat_005(backend, dtype="complex128", seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)

    _vals = backend.randn((2,1,2), dtype=dtype, seed=seed) 
    vals  = _vals[:,0,:]
    xvals = ar.asarray(_vals)
 
    pos  = (slice(0,2), 1, slice(2,4))
    xpos = (
            ar.asarray([[[0,0]], [[1,1]]], backend=backend), 
            ar.asarray([[[1,1]], [[1,1]]], backend=backend),
            ar.asarray([[[2,3]], [[2,3]]], backend=backend),
           )

    dense      = backend.zeros(shape, dtype=dtype)
    dense[pos] = vals[:,:]

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
    )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, xpos, xvals)

    valinds = (
               x.inds[0].resized(0,2),  
               x.inds[2].resized(2,4),
              ) 

    return SparseGradData(
                          grad,     space,          
                          x.tensor, x.array,  x.data,
                          vals,     xvals,    valinds,           
                          pos,      xpos,             
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




def sparsegrad_dat_006(backend, dtype="complex128", seed=1):

    backend  = backends.get(backend) 
    indnames = "ijk"
    shape    = (2,3,4)

    _vals = backend.randn((2,1,2), dtype=dtype, seed=seed) 
    vals  = _vals[:,0,:]
    xvals = ar.asarray(_vals)
 
    pos  = (slice(None), 1, slice(2,4))
    xpos = (
            ar.asarray([[[0,0]], [[1,1]]], backend=backend), 
            ar.asarray([[[1,1]], [[1,1]]], backend=backend),
            ar.asarray([[[2,3]], [[2,3]]], backend=backend),
           )

    dense      = backend.zeros(shape, dtype=dtype)
    dense[pos] = vals[:,:]

    def densefun(shape, dtype):
        return dense

    x = tensor_dat(densefun)(
           backend.name(), indnames, shape, dtype=dtype
    )
    w = data.arrayspace_dat(backend.name(), shape, dtype)

    space = tn.TensorSpace(w.space, x.inds)
    grad  = tn.SparseGrad(space, xpos, xvals)

    valinds = (
               x.inds[0].resized(0,2),  
               x.inds[2].resized(2,4),
              ) 

    return SparseGradData(
                          grad,     space,          
                          x.tensor, x.array,  x.data,
                          vals,     xvals,    valinds,           
                          pos,      xpos,             
                          x.inds,   x.shape,  dtype,  x.backend,  {}
                         )




# --- Tensor data ----------------------------------------------------------- #

TensorData = collections.namedtuple("TensorData", [
                "tensor",  "array",  "data", 
                "inds",    "shape",  "dtype", 
                "backend", "opts",
             ])




def tensor_dat(datafun):

    def wrap(backend, indnames, shape, **opts):

        v = indices_dat(indnames, shape)
        w = data.array_dat(datafun)(backend, shape, **opts)

        tensor = tn.TensorGen(w.array, v.inds)

        return TensorData(
           tensor,    w.array,  w.data, 
           v.inds,    w.shape,  w.array.dtype, 
           w.backend, w.opts
        )

    return wrap




# --- N-Tensor data --------------------------------------------------------- #

NTensorData = collections.namedtuple("NTensorData", [
                "tensors", "arrays", "datas", 
                "inds", "shapes", "backend", "opts",
             ])




def ntensor_dat(datafun):

    def wrap(backend, indnames, shapes, **opts):

        seed = opts.get("seed",   0)
        v    = nindices_dat(indnames, shapes)

        ws = []
        ts = []

        for i in range(len(shapes)):

            w = data.array_dat(datafun)(backend, shapes[i], seed=seed+i+1)
            ws.append(w)

            t = tn.TensorGen(w.array, v.inds.map(*indnames[i]))
            ts.append(t)  

        tensors = ts
        arrays  = [w.array for w in ws]
        datas   = [w.data  for w in ws]
        
        return NTensorData(
           tensors, arrays, datas, 
           v.inds, shapes, backend, opts
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

    space = tn.TensorSpace(w.space, v.inds)

    return TensorSpaceData(
              space,   w.space, v.inds, 
              w.shape, w.dtype, w.backend
           ) 




def tensorspace_dat_001(backend):

    return tensorspace_dat(
              backend, "ijk", (2,3,4), "complex128"
           )








