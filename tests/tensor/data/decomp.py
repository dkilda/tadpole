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
import tests.array.data   as ardata


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

    x        = ar.randn(shape, dtype="complex128", seed=1)
    U, _, VH = ar.svd(x)

    array = ar.dot(U, ar.dot(ar.diag(S), VH))

    return SvdData(
              array, U, S, VH, 
              size, shape, backend,
           )

    


