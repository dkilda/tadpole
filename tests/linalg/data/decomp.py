#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid
import tadpole.linalg   as la

import tadpole.array.backends as backends

import tests.linalg.fakes as fake
import tests.linalg.data  as data


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




# --- Truncation data ------------------------------------------------------- #

TruncData = collections.namedtuple("TruncData", [
               "array", "U", "S", "V", 
               "rank", "shape", "backend",
            ])




def svd_trunc_dat(backend):

    backend = backends.get(backend)

    rank  = 10
    shape = (13,10)

    data = backend.asarray([
              8.84373474e-01, 4.46849762e-01, 1.18639293e-01, 6.40245186e-02,
              5.13656787e-03, 2.75924093e-03, 6.51826227e-04, 6.91792508e-05,
              7.41402228e-06, 9.25687086e-07
           ])
    S = ar.asarray(data, backend=backend)

    x        = ar.randn(shape, dtype="complex128", seed=1, backend=backend)
    U, _, VH = ar.svd(x)

    array = ar.dot(U, ar.dot(ar.diag(S), VH))

    return TruncData(
              array, U, S, VH, 
              rank, shape, backend,
           )




# --- Sample array data for truncated decompositions ------------------------ #

def randn_decomp_000(shape, dtype="complex128", seed=1, backend=None):

    if shape != (12,10):
       raise ValueError(
          f"randn_decomp_001: "
          f"this data generator only supports "
          f"shape (12,10) but not {shape}"
       )

    backend = backends.get(backend)
    rank    = 10
    shape   = (12,10)

    S = backend.asarray([
              8.84373474e-01, 4.46849762e-01, 1.18639293e-01, 6.40245186e-02,
              5.13656787e-03, 2.75924093e-03, 6.51826227e-04, 6.91792508e-05,
              7.41402228e-06, 9.25687086e-07
           ])

    x       = backend.randn(shape, dtype="complex128", seed=seed)
    U, _, V = backend.svd(x)

    return backend.dot(U, backend.dot(backend.diag(S), V))






# --- Decomposition input data ---------------------------------------------- #

DecompInputData = collections.namedtuple("DecompInputData", [
                     "inds",  "shape",  
                     "linds", "lshape", "lsize", "laxes", 
                     "rinds", "rshape", "rsize", "raxes",
                     "sind",  "srank", 
                  ])




def decomp_input_000():

    return DecompInputData(

       inds   = "ijk",     
       shape  = (6,10,2),

       linds  = "ik",
       lshape = (6,2),
       lsize  = 12,
       laxes  = (0,2),

       rinds  = "j",
       rshape = (10,),
       rsize  = 10,
       raxes  = (1,),

       sind   = "s",
       srank  = 10,
    )




def decomp_input_001():

    return DecompInputData(

       inds   = "ijk",     
       shape  = (2,3,4),

       linds  = "ik",
       lshape = (2,4),
       lsize  = 8,
       laxes  = (0,2),

       rinds  = "j",
       rshape = (3,),
       rsize  = 3,
       raxes  = (1,),

       sind   = "s",
       srank  = 3,
    )




def decomp_input_002():

    return DecompInputData(

       inds   = "ijk",     
       shape  = (2,8,4),

       linds  = "ik",
       lshape = (2,4),
       lsize  = 8,
       laxes  = (0,2),

       rinds  = "j",
       rshape = (8,),
       rsize  = 8,
       raxes  = (1,),

       sind   = "s",
       srank  = 8,
    )




# --- Hidden-rank decomposition data ---------------------------------------- #

HiddenDecompData = collections.namedtuple("HiddenDecompData", [
                      "xtensor", "ltensor", "rtensor", 
                      "xarray",  "larray",  "rarray",
                      "xmatrix", "lmatrix", "rmatrix", 
                      "xinds",   "linds",   "rinds",  "sind",  
                      "shape",   "lshape",  "rshape", "srank",
                      "backend", 
                   ])




def hidden_decomp_data(datafun):

    def wrap(method, backend, decomp_input_dat, **opts):

        # --- Input --- #

        w = decomp_input_dat 
        v = data.indices_dat(w.inds + w.sind, w.shape + (w.srank,))
        x = data.array_dat(datafun)(backend, (w.lsize, w.rsize), **opts)  

        # --- Matrix decomp --- #

        xmatrix = x.array
        xarray  = ar.reshape(xmatrix,  (*w.lshape, *w.rshape))  
        xarray  = ar.transpose(xarray, v.inds.axes(*(w.linds + w.rinds)))      

        larray, rarray = {
                          "qr": ar.qr,
                          "lq": ar.lq,
                         }[method](xmatrix)

        lmatrix = larray
        rmatrix = rarray

        # --- Create tensors --- #

        larray = ar.reshape(larray, (*w.lshape, w.srank))
        rarray = ar.reshape(rarray, (w.srank, *w.rshape))

        sind  = v.inds.map(*w.sind)[0]
        linds = v.inds.map(*w.linds)     
        rinds = v.inds.map(*w.rinds)  
        xinds = v.inds.map(*w.inds)

        ltensor = tn.TensorGen(larray, (*linds, sind))
        rtensor = tn.TensorGen(rarray, (sind,   *rinds))
        xtensor = tn.TensorGen(xarray, xinds)

        return HiddenDecompData(
                  xtensor,  ltensor,   rtensor, 
                  xarray,   larray,    rarray,
                  xmatrix,  lmatrix,   rmatrix,   
                  xinds,    linds,     rinds,     sind,
                  w.shape,  w.lshape,  w.rshape,  w.srank,
                  x.backend,
               )

    return wrap




# --- QR decomposition data ------------------------------------------------- #

def qr_tensor_dat(decomp_input):

    def wrap(datafun, backend, **opts):

        return hidden_decomp_data(datafun)(
                  "qr", backend, decomp_input(), **opts
               )
    return wrap




# --- LQ decomposition data ------------------------------------------------- #

def lq_tensor_dat(decomp_input):

    def wrap(datafun, backend, **opts):

        return hidden_decomp_data(datafun)(
                  "lq", backend, decomp_input(), **opts
               )
    return wrap




# --- Explicit-rank decomposition data ---------------------------------------- #

ExplicitDecompData = collections.namedtuple("ExplicitDecompData", [
                        "xtensor", "ltensor", "rtensor", "stensor",
                        "xarray",  "larray",  "rarray",  "sarray",
                        "xmatrix", "lmatrix", "rmatrix", "smatrix",
                        "xinds",   "linds",   "rinds",   "sind",  
                        "shape",   "lshape",  "rshape",  "srank",
                        "backend", "error", 
                     ])




def explicit_decomp_data(datafun):

    def wrap(method, backend, decomp_input_dat, trunc=la.TruncNull(), **opts):

        # --- Input --- #

        w = decomp_input_dat 
        x = data.array_dat(datafun)(backend, (w.lsize, w.rsize), **opts) 

        # --- Matrix decomp --- #

        xmatrix = x.array
        if method == "eigh":
           xmatrix = xmatrix + ar.transpose(ar.conj(xmatrix), (1,0))
              
        larray, sarray, rarray = {
                                  "svd":  ar.svd,
                                  "eig":  ar.eig,
                                  "eigh": ar.eigh,
                                 }[method](xmatrix)

        error                  = trunc.error(sarray)
        srank                  = trunc.rank(sarray)
        larray, sarray, rarray = trunc.apply(larray, sarray, rarray)

        lmatrix = larray
        rmatrix = rarray
        smatrix = sarray

        # --- Create tensors --- #

        v = data.indices_dat(w.inds + w.sind, w.shape + (srank,))

        xarray = ar.reshape(xmatrix,  (*w.lshape, *w.rshape))  
        xarray = ar.transpose(xarray, v.inds.axes(*(w.linds + w.rinds)))  

        larray = ar.reshape(larray, (*w.lshape, srank))
        rarray = ar.reshape(rarray, ( srank,    *w.rshape))

        sind  = v.inds.map(*w.sind)[0]
        linds = v.inds.map(*w.linds)     
        rinds = v.inds.map(*w.rinds)  
        xinds = v.inds.map(*w.inds)

        ltensor = tn.TensorGen(larray, (*linds, sind))
        rtensor = tn.TensorGen(rarray, (sind,   *rinds))
        stensor = tn.TensorGen(sarray, (sind,))
        xtensor = tn.TensorGen(xarray, xinds)

        return ExplicitDecompData(
                  xtensor,   ltensor,   rtensor,   stensor,
                  xarray,    larray,    rarray,    sarray,
                  xmatrix,   lmatrix,   rmatrix,   smatrix,                  
                  xinds,     linds,     rinds,     sind,
                  w.shape,   w.lshape,  w.rshape,  srank,
                  x.backend, error,
               )

    return wrap




# --- SVD decomposition data ------------------------------------------------ #

def svd_tensor_dat(decomp_input):

    def wrap(datafun, backend, **opts):

        return explicit_decomp_data(datafun)(
                  "svd", backend, decomp_input(), **opts
               )
    return wrap




# --- Eigenvalue decomposition data ----------------------------------------- #

def eig_tensor_dat(decomp_input):

    def wrap(datafun, backend, **opts):

        return explicit_decomp_data(datafun)(
                  "eig", backend, decomp_input(), **opts
               )
    return wrap




# --- Hermitian eigenvalue decomposition data ------------------------------- #

def eigh_tensor_dat(decomp_input):

    def wrap(datafun, backend, **opts):

        return explicit_decomp_data(datafun)(
                  "eigh", backend, decomp_input(), **opts
               )
    return wrap




