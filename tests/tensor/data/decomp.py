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




# --- Decomposition input data ---------------------------------------------- #

DecompInputData = collections.namedtuple("DecompInputData", [
                     "inds",  "shape",  
                     "linds", "lshape", "lsize", "laxes", 
                     "rinds", "rshape", "rsize", "raxes",
                     "sind",  "srank", 
                  ])




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

        w = decomp_input_dat 
        v = data.indices_dat(w.inds + w.sind, w.shape + (w.srank,))
        x = data.array_dat(datafun)(backend, (w.lsize, w.rsize), **opts)  

        xmatrix = x.array
        xarray  = ar.reshape(xmatrix,   (*w.lshape, *w.rshape))  
        xarray  = ar.transpose(xarray, v.inds.axes(*(w.linds + w.rinds)))      

        larray, rarray = {
                          "qr": ar.qr,
                          "lq": ar.lq,
                         }[method](xmatrix)

        lmatrix = larray
        rmatrix = rarray

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

def qr_tensor_dat(datafun, backend, **opts):

    w = DecompInputData(

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

    return hidden_decomp_data(datafun)("qr", backend, w, **opts)




# --- LQ decomposition data ------------------------------------------------- #

def lq_tensor_dat(datafun, backend, **opts):

    w = DecompInputData(

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

    return hidden_decomp_data(datafun)("lq", backend, w, **opts)




# --- Explicit-rank decomposition data ---------------------------------------- #

ExplicitDecompData = collections.namedtuple("ExplicitDecompData", [
                        "xtensor", "ltensor", "rtensor", "stensor",
                        "xarray",  "larray",  "rarray",  "sarray",
                        "xmatrix", "lmatrix", "rmatrix", "smatrix",
                        "xinds",   "linds",   "rinds",   "sind",  
                        "shape",   "lshape",  "rshape",  "srank",
                        "backend", 
                     ])




def explicit_decomp_data(datafun):

    def wrap(method, backend, decomp_input_dat, **opts):

        w = decomp_input_dat 
        v = data.indices_dat(w.inds + w.sind, w.shape + (w.srank,))
        x = data.array_dat(datafun)(backend, (w.lsize, w.rsize), **opts) 

        xmatrix = x.array
        if method == "eigh":
           xmatrix = ar.add(xmatrix, ar.transpose(ar.conj(xmatrix), (1,0))) 
           
        xarray  = ar.reshape(xmatrix,   (*w.lshape, *w.rshape))  
        xarray  = ar.transpose(xarray, v.inds.axes(*(w.linds + w.rinds)))      

        larray, sarray, rarray = {
                                  "svd":  ar.svd,
                                  "eig":  ar.eig,
                                  "eigh": ar.eigh,
                                 }[method](xmatrix)

        lmatrix = larray
        rmatrix = rarray
        smatrix = sarray

        larray = ar.reshape(larray, (*w.lshape, w.srank))
        rarray = ar.reshape(rarray, (w.srank, *w.rshape))

        sind  = v.inds.map(*w.sind)[0]
        linds = v.inds.map(*w.linds)     
        rinds = v.inds.map(*w.rinds)  
        xinds = v.inds.map(*w.inds)

        ltensor = tn.TensorGen(larray, (*linds, sind))
        rtensor = tn.TensorGen(rarray, (sind,   *rinds))
        stensor = tn.TensorGen(sarray, (sind,))
        xtensor = tn.TensorGen(xarray, xinds)

        return ExplicitDecompData(
                  xtensor,  ltensor,   rtensor,   stensor,
                  xarray,   larray,    rarray,    sarray,
                  xmatrix,  lmatrix,   rmatrix,   smatrix,
                  xinds,    linds,     rinds,     sind,
                  w.shape,  w.lshape,  w.rshape,  w.srank,
                  x.backend, 

               )

    return wrap




# --- SVD decomposition data ------------------------------------------------ #

def svd_tensor_dat(datafun, backend, **opts):

    w = DecompInputData(

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

    return explicit_decomp_data(datafun)("svd", backend, w, **opts)




# --- Eigenvalue decomposition data ----------------------------------------- #

def eig_tensor_dat(datafun, backend, **opts):

    w = DecompInputData(

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

    return explicit_decomp_data(datafun)("eig", backend, w, **opts)




# --- Hermitian eigenvalue decomposition data ------------------------------- #

def eigh_tensor_dat(datafun, backend, **opts):

    w = DecompInputData(

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

    return explicit_decomp_data(datafun)("eigh", backend, w, **opts)





















