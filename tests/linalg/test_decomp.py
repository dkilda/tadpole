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
import tadpole.linalg   as la

import tadpole.array.backends as backends
import tadpole.tensor.engine  as tne 

import tests.linalg.fakes as fake
import tests.linalg.data  as data

from tests.common import (
   available_backends,
)

from tadpole.linalg.decomp import (
   SIndexFun, 
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




###############################################################################
###                                                                         ###
###  Helpers: indexing and other logic needed for a tensor decomposition    ### 
###                                                                         ###
###############################################################################


# --- Functor that creates the s-index emerging from decomposition ---------- #

class TestSIndexFun:

   @pytest.mark.parametrize("tag, size", [
      ["i", 3],
   ])   
   def test_ind(self, tag, size):

       sind = SIndexFun(tag)
       out  = sind(size)

       assert len(out) == size
       assert out.all(tag) 


   @pytest.mark.parametrize("tag, size", [
      ["i", 3],
   ])   
   def test_ind_001(self, tag, size):

       sind = SIndexFun(tag)
       out  = sind(size)
       out1 = sind(size)

       assert out == out1


   @pytest.mark.parametrize("tag, size", [
      ["i", 3],
   ])   
   def test_ind_002(self, tag, size):

       sind = SIndexFun(tag)
       out  = sind(size)

       try:
          out1 = sind(size+1)
       except ValueError:
          assert True
       else:
          assert False




###############################################################################
###                                                                         ###
###  Tensor decomposition engine and operator                               ###
###                                                                         ###
###############################################################################


# --- Tensor decomposition operator ----------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTensorDecomp:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Explicit-rank decompositions with truncation --- #

   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   @pytest.mark.parametrize("trunc", [
      la.TruncNull(),
      la.TruncRel(1e-5),
      la.TruncRelSum2(1e-5),
      la.TruncRel(1e-5, 7),
      la.TruncRelSum2(1e-5, 7),
   ])
   def test_svd_trunc(self, alignment, trunc):

       w = data.svd_tensor_dat(data.decomp_input_000)(
              data.randn_decomp_000, self.backend, trunc=trunc
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       U, S, V, error = la.svd(w.xtensor, **inds, sind="s", trunc=trunc)

       sind, = tn.overlap_inds(U, S)
       U1    = tn.reindex(w.ltensor, {w.sind: sind})
       S1    = tn.reindex(w.stensor, {w.sind: sind})
       V1    = tn.reindex(w.rtensor, {w.sind: sind})

       x  = tn.contract(U,  S,  V,  product=w.xinds)
       x1 = tn.contract(U1, S1, V1, product=w.xinds)

       assert U.space() == U1.space()
       assert S.space() == S1.space()
       assert V.space() == V1.space()

       assert ar.allclose(error, w.error)
       assert tn.allclose(S, S1)
       assert tn.allclose(x, x1)


   # --- Explicit-rank decompositions --- #

   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_svd(self, decomp_input, alignment):

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       U, S, V, error = la.svd(w.xtensor, **inds, sind="s")

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


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_eig(self, decomp_input, alignment):

       w = data.eig_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       V, S = la.eig(w.xtensor, **inds, sind="s")

       sind, = tn.overlap_inds(V, S)
       V1    = tn.reindex(w.ltensor, {w.sind: sind})
       S1    = tn.reindex(w.stensor, {w.sind: sind})

       assert V.space() == V1.space()
       assert S.space() == S1.space()

       i = IndexGen("i", tid.sizeof(*w.linds))

       out1 = tn.contract(V, S, product=Indices(*w.linds, sind))
       out2 = tn.contract(
                 tn.fuse(w.xtensor, {w.rinds: i}), 
                 tn.fuse(V,         {w.linds: i})
              )

       assert tn.allclose(out1, out2)
       assert tn.allclose(S, S1)


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_eigh(self, decomp_input, alignment):

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       V, S = la.eigh(w.xtensor, **inds, sind="s")

       sind, = tn.overlap_inds(V, S)
       V1    = tn.reindex(w.ltensor, {w.sind: sind})
       S1    = tn.reindex(w.stensor, {w.sind: sind})

       assert V.space() == V1.space()
       assert S.space() == S1.space()

       i = IndexGen("i", tid.sizeof(*w.linds))

       out1 = tn.contract(V, S, product=Indices(*w.linds, sind))
       out2 = tn.contract(
                 tn.fuse(w.xtensor, {w.rinds: i}), 
                 tn.fuse(V,         {w.linds: i})
              )

       assert tn.allclose(out1, out2)
       assert tn.allclose(S, S1)

       
   # --- Hidden-rank decompositions --- #

   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_qr(self, decomp_input, alignment):

       w = data.qr_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       Q, R = la.qr(w.xtensor, **inds, sind="s")

       sind, = tn.overlap_inds(Q, R)
       Q1    = tn.reindex(w.ltensor, {w.sind: sind})
       R1    = tn.reindex(w.rtensor, {w.sind: sind})

       assert Q.space() == Q1.space()
       assert R.space() == R1.space()

       assert tn.allclose(Q, Q1)
       assert tn.allclose(R, R1)


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_lq(self, decomp_input, alignment):

       w = data.lq_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       L, Q = la.lq(w.xtensor, **inds, sind="s")

       sind, = tn.overlap_inds(L, Q)
       L1    = tn.reindex(w.ltensor, {w.sind: sind})
       Q1    = tn.reindex(w.rtensor, {w.sind: sind})

       assert L.space() == L1.space()
       assert Q.space() == Q1.space()

       assert tn.allclose(L, L1)
       assert tn.allclose(Q, Q1)




