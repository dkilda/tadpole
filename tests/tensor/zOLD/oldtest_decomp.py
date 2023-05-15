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


from tests.common import (
   available_backends,
)


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


# --- Left alignment -------------------------------------------------------- #

class TestLeftAlignment:

   @pytest.mark.parametrize("shape, inds, partial, rest", [
      [(2,3,4), "ijk", "ik", "j"],
   ])
   def test_linds(self, shape, inds, partial, rest):

       v = data.indices_dat(inds, shape)
       x = decomp.LeftAlignment(partial)

       assert x.linds(v.inds) == Indices(*v.inds.map(*partial))


   @pytest.mark.parametrize("shape, inds, partial, rest", [
      [(2,3,4), "ijk", "ik", "j"],
   ])
   def test_rinds(self, shape, inds, partial, rest):

       v = data.indices_dat(inds, shape)
       x = decomp.LeftAlignment(partial)

       assert x.rinds(v.inds) == Indices(*v.inds.map(*rest))




# --- Right alignment -------------------------------------------------------- #

class TestRightAlignment:

   @pytest.mark.parametrize("shape, inds, partial, rest", [
      [(2,3,4), "ijk", "ik", "j"],
   ])
   def test_linds(self, shape, inds, partial, rest):

       v = data.indices_dat(inds, shape)
       x = decomp.RightAlignment(partial)

       assert x.linds(v.inds) == Indices(*v.inds.map(*rest))


   @pytest.mark.parametrize("shape, inds, partial, rest", [
      [(2,3,4), "ijk", "ik", "j"],
   ])
   def test_rinds(self, shape, inds, partial, rest):

       v = data.indices_dat(inds, shape)
       x = decomp.RightAlignment(partial)

       assert x.rinds(v.inds) == Indices(*v.inds.map(*partial))



# --- Link between partitions ----------------------------------------------- #

class TestLink:

   @pytest.mark.parametrize("tag, size", [
      ["i", 3],
   ])   
   def test_ind(self, tag, size):

       link = decomp.Link(tag)
       out  = link.ind(size)

       assert len(out) == size
       assert out.all(tag) 


   @pytest.mark.parametrize("tag, size", [
      ["i", 3],
   ])   
   def test_ind_001(self, tag, size):

       link = decomp.Link(tag)
       out  = link.ind(size)
       out1 = link.ind(size)

       assert out == out1


   @pytest.mark.parametrize("tag, size", [
      ["i", 3],
   ])   
   def test_ind_002(self, tag, size):

       link = decomp.Link(tag)
       out  = link.ind(size)

       try:
          out1 = link.ind(size+1)
       except ValueError:
          assert True
       else:
          assert False




# --- Partition ------------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestPartition:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   def test_aligndata(self, decomp_input):

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend
           )
       x = decomp.Partition(
                            Indices(*w.xinds), 
                            Indices(*w.linds), 
                            Indices(*w.rinds), 
                            decomp.Link("s")
                           )

       assert ar.allclose(x.aligndata(w.xarray), w.xmatrix)


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   def test_ltensor(self, decomp_input):

       link = decomp.Link("s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend
           )
       x = decomp.Partition(
                            Indices(*w.xinds), 
                            Indices(*w.linds), 
                            Indices(*w.rinds), 
                            link,
                           )

       sind = link.ind(len(w.sind))

       out = x.ltensor(w.lmatrix)
       ans = tn.reindex(w.ltensor, {w.sind: sind})

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   def test_rtensor(self, decomp_input):

       link = decomp.Link("s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend
           )
       x = decomp.Partition(
                            Indices(*w.xinds), 
                            Indices(*w.linds), 
                            Indices(*w.rinds), 
                            link,
                           )

       sind = link.ind(len(w.sind))

       out = x.rtensor(w.rmatrix)
       ans = tn.reindex(w.rtensor, {w.sind: sind})

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   def test_stensor(self, decomp_input):

       link = decomp.Link("s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend
           )
       x = decomp.Partition(
                            Indices(*w.xinds), 
                            Indices(*w.linds), 
                            Indices(*w.rinds), 
                            link,
                           )

       sind = link.ind(len(w.sind))

       out = x.stensor(w.smatrix)
       ans = tn.reindex(w.stensor, {w.sind: sind})

       assert tn.allclose(out, ans)




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
      "left", 
      "right",
   ])
   @pytest.mark.parametrize("trunc", [
      tn.TruncNull(),
      tn.TruncRel(1e-5),
      tn.TruncRelSum2(1e-5),
      tn.TruncRel(1e-5, 7),
      tn.TruncRelSum2(1e-5, 7),
   ])
   def test_svd_trunc(self, alignment, trunc):

       w = data.svd_tensor_dat(data.decomp_input_000)(
              data.randn_decomp_000, self.backend, trunc=trunc
           )

       U, S, V, error = tn.svd(
                           w.xtensor, 
                           {"left": w.linds, "right": w.rinds}[alignment], 
                           alignment, 
                           "s",
                           trunc,
                        )

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
      "left", 
      "right",
   ])
   def test_svd(self, decomp_input, alignment):

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

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


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   @pytest.mark.parametrize("alignment", [
      "left", 
      "right",
   ])
   def test_eig(self, decomp_input, alignment):

       w = data.eig_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       U, S, V, error = tn.eig(
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


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "left", 
      "right",
   ])
   def test_eigh(self, decomp_input, alignment):

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       U, S, V, error = tn.eigh(
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

   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   @pytest.mark.parametrize("alignment", [
      "left", 
      "right",
   ])
   def test_qr(self, decomp_input, alignment):

       w = data.qr_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

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


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
   ])
   @pytest.mark.parametrize("alignment", [
      "left", 
      "right",
   ])
   def test_lq(self, decomp_input, alignment):

       w = data.lq_tensor_dat(decomp_input)(
              data.randn, self.backend
           )

       L, Q = tn.lq(
                    w.xtensor, 
                    {"left": w.linds, "right": w.rinds}[alignment], 
                    alignment, 
                    "s"
                   )

       sind, = tn.overlap_inds(L, Q)
       L1    = tn.reindex(w.ltensor, {w.sind: sind})
       Q1    = tn.reindex(w.rtensor, {w.sind: sind})

       assert L.space() == L1.space()
       assert Q.space() == Q1.space()

       assert tn.allclose(L, L1)
       assert tn.allclose(Q, Q1)















