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

import tadpole.array.backends   as backends
import tadpole.tensor.reduction as redu
import tadpole.tensor.engine    as tne 

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
###  Tensor reduction engine and operator                                   ###
###                                                                         ###
###############################################################################


# --- Tensor reduction operator --------------------------------------------- #  

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTensorReduce:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Value methods --- #

   @pytest.mark.parametrize("winds, wshape, inds", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_allof(self, winds, wshape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.allof(w.tensor)
            ans = ar.allof(w.array)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.allof(w.tensor, inds)
            ans = ar.allof(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans,  w.inds.map(*output_inds))

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("winds, wshape, inds", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_anyof(self, winds, wshape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.anyof(w.tensor)
            ans = ar.anyof(w.array)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.anyof(w.tensor, inds)
            ans = ar.anyof(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans,  w.inds.map(*output_inds))

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("winds, wshape, inds", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_amax(self, winds, wshape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.amax(w.tensor)
            ans = ar.amax(w.array)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.amax(w.tensor, inds)
            ans = ar.amax(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans, w.inds.map(*output_inds))

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("winds, wshape, inds", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_amin(self, winds, wshape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.amin(w.tensor)
            ans = ar.amin(w.array)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.amin(w.tensor, inds)
            ans = ar.amin(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans, w.inds.map(*output_inds))

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("winds, wshape, inds", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_count_nonzero(self, winds, wshape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.count_nonzero(w.tensor)
            ans = ar.count_nonzero(w.array)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.count_nonzero(w.tensor, inds)
            ans = ar.count_nonzero(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans,          w.inds.map(*output_inds))

       assert tn.allclose(out, ans)


   # --- Shape methods --- #

   @pytest.mark.parametrize("winds, wshape, inds", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_sumover(self, winds, wshape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.sumover(w.tensor)
            ans = ar.sumover(w.array)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.sumover(w.tensor, inds)
            ans = ar.sumover(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans,    w.inds.map(*output_inds))

       assert tn.allclose(out, ans)


   # --- Linear algebra methods --- #

   @pytest.mark.parametrize("winds, wshape, inds, order", [
      ["ijk", (2,3,4), None, None],
      ["ijk", (2,3,4), "j",  None],
      ["ijk", (2,3,4), "j",  0],
      ["ijk", (2,3,4), "ij", None],
      ["ijk", (2,3,4), "ij", "fro"],
      ["ijk", (2,3,4), "ij", "nuc"],
      ["ijk", (2,3,4), "ij",  1],
      ["ijk", (2,3,4), "ij",  2],
      ["ijk", (2,3,4), "ij", -1],
      ["ijk", (2,3,4), "ij", -2],
   ])
   def test_norm(self, winds, wshape, inds, order):

       w = data.tensor_dat(data.randn)(
              self.backend, winds, wshape
           )

       if   inds is None:
            out = tn.norm(w.tensor, order=order)
            ans = ar.norm(w.array,  order=order)
            ans = tn.TensorGen(ans)

       else:
            output_inds = "".join(util.complement(winds, inds)) 

            out = tn.norm(w.tensor, inds,               order=order)
            ans = ar.norm(w.array,  w.inds.axes(*inds), order=order)
            ans = tn.TensorGen(ans, w.inds.map(*output_inds))

       assert tn.allclose(out, ans)




