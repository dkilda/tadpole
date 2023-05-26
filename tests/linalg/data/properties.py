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




# --- Property input data ---------------------------------------------- #

PropertyInputData = collections.namedtuple("PropertyInputData", [
                     "inds",  "shape",  
                     "linds", "lshape", "lsize", "laxes", 
                     "rinds", "rshape", "rsize", "raxes",
                  ])




def property_input_001():

    return PropertyInputData(

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
    )




def property_input_002():

    return PropertyInputData(

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
    )




def property_input_003():

    return PropertyInputData(

       inds   = "ijk",     
       shape  = (2,3,4),

       linds  = "j",
       lshape = (3,),
       lsize  = 3,
       laxes  = (1,),

       rinds  = "ik",
       rshape = (2,4),
       rsize  = 8,
       raxes  = (0,2),
    )




# --- Property data ---------------------------------------- #

PropertyData = collections.namedtuple("PropertyData", [
                  "tensor", "array",  "matrix", 
                  "inds",   "linds",  "rinds",   
                  "shape",  
                  "lshape", "rshape",  
                  "lsize",  "rsize",
                  "backend", 
               ])




def property_linalg_dat(property_input):

    def wrap(datafun, backend, **opts):

        w = property_input() 
        v = data.indices_dat(w.inds, w.shape)
        x = data.array_dat(datafun)(backend, (w.lsize, w.rsize), **opts)  

        xmatrix = x.array
        xarray  = ar.reshape(xmatrix,  (*w.lshape, *w.rshape))  
        xarray  = ar.transpose(xarray, v.inds.axes(*(w.linds + w.rinds)))      

        xinds = v.inds.map(*w.inds)
        linds = v.inds.map(*w.linds)     
        rinds = v.inds.map(*w.rinds)  

        xtensor = tn.TensorGen(xarray, xinds)

        return PropertyData(
                  xtensor,  xarray,    xmatrix,     
                  xinds,    linds,     rinds,     
                  w.shape,  
                  w.lshape, w.rshape,  
                  w.lsize,  w.rsize,
                  x.backend,
               )

    return wrap




