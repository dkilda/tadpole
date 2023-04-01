#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

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




# --- Decomp data ----------------------------------------------------------- #

SvdData = collections.namedtuple("SvdData", [
             "array", "U", "S", "VH", 
             "size", "shape", "backend",
          ])




def svd_dat():

    backend = backends.get(backend)

    size  = 10
    shape = (13,10)

    data = backend.asarray([
              8.84373474e-01, 4.46849762e-01, 1.18639293e-01, 6.40245186e-02,
              5.13656787e-03, 2.75924093e-03, 6.51826227e-04, 6.91792508e-05,
              7.41402228e-06, 9.25687086e-07
           ])
    S = Array(backend, data)

    x        = ar.randn(shape, dtype="complex128", seed=1, backend=backend)
    U, _, VH = ar.svd(x)

    array = ar.dot(U, ar.dot(ar.diag(S), VH))

    return SvdData(
              array, U, S, VH, 
              size, shape, backend,
           )












DecompInputData = collections.namedtuple("DecompInputData", [
                     "inds",  "shape",  
                     "linds", "lshape", "lsize", "laxes", 
                     "rinds", "rshape", "rsize", "raxes",
                     "sind",  "srank", 
                  ])



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

        xarray = x.array
        xarray = ar.reshape(xarray,   (*w.lshape, *w.rshape))  
        xarray = ar.transpose(xarray, v.inds.axes(*(w.linds + w.rinds)))      

        larray, rarray = {
                          "qr": ar.qr,
                          "lq": ar.lq,
                         }[method](x.array)

        xmatrix = x.array
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

        xarray = x.array
        xarray = ar.reshape(xarray,   (*w.lshape, *w.rshape))  
        xarray = ar.transpose(xarray, v.inds.axes(*(w.linds + w.rinds)))      

        larray, sarray, rarray = {
                                  "svd":  ar.svd,
                                  "eig":  ar.eig,
                                  "eigh": ar.eigh,
                                 }[method](x.array)

        xmatrix = x.array
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



















    


