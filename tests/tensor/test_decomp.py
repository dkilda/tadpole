#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.array.backends as backends
import tadpole.tensor.decomp  as decomp
import tadpole.tensor.engine  as tne 

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




###############################################################################
###                                                                         ###
###  The logic of index partitioning: defines the alignment by left/right   ### 
###  indices and a link between the left/right partitions.                  ###
###                                                                         ###
###############################################################################


# --- Alignment ------------------------------------------------------------- #


# --- Link between partitions ----------------------------------------------- #



# --- Partition ------------------------------------------------------------- #




###############################################################################
###                                                                         ###
###  Tensor decomposition engine and operator                               ###
###                                                                         ###
###############################################################################


# --- Tensor decomposition operator ----------------------------------------- #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTensorDecomp:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Explicit-rank decompositions --- #

   @pytest.mark.parametrize("svddat", [
      data.svd_tensor_dat,
   ])
   @pytest.mark.parametrize("alignment", [
      "left", 
      "right",
   ])
   def test_svd(self, svddat, alignment):

       w = svddat(data.randn, self.backend)

       U, S, V, error = tn.svd(
                           w.xtensor, 
                           {"left": w.linds, "right": w.rinds}[alignment], 
                           alignment, 
                           "s"
                        )

       sind, = tn.overlap_inds(U, S)
       U1    = tn.reindex(w.ltensor, {w.sind: sind})
       S1    = tn.reindex(w.stensor, {w.sind: sind})
       V1    = tn.reindex(w.rtensor, {w.sind: sind})

       assert U.space() == U1.space()
       assert S.space() == S1.space()
       assert V.space() == V1.space()

       assert tn.allclose(S, S1)
       assert ar.allclose(error, 0)
       assert tn.allclose(tn.contract(U, S, V, product=w.xinds), w.xtensor)

       
   # --- Hidden-rank decompositions --- #

   @pytest.mark.parametrize("qrdat", [
      data.qr_tensor_dat,
   ])
   @pytest.mark.parametrize("alignment", [
      "left", 
      "right",
   ])
   def test_qr(self, qrdat, alignment):

       w = qrdat(data.randn, self.backend)

       Q, R = tn.qr(
                    w.xtensor, 
                    {"left": w.linds, "right": w.rinds}[alignment], 
                    alignment, 
                    "s"
                   )

       sind, = tn.overlap_inds(Q, R)
       Q1    = tn.reindex(w.ltensor, {w.sind: sind})
       R1    = tn.reindex(w.rtensor, {w.sind: sind})

       assert Q.space() == Q1.space()
       assert R.space() == R1.space()

       assert tn.allclose(Q, Q1)
       assert tn.allclose(R, R1)
















