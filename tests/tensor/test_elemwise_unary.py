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

import tadpole.array.backends        as backends
import tadpole.tensor.elemwise_unary as tnu
import tadpole.tensor.engine         as tne 

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


# pytest util autodiff global array tensor

###############################################################################
###                                                                         ###
###  Tensor unary elementwise engine and operator                           ###
###                                                                         ###
###############################################################################


# --- Helper: unary typecast unary ------------------------------------------ #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTypecastUnary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("fundat",  [
       data.unary_wrappedfun_dat_001,
       data.unary_wrappedfun_dat_002,
   ])
   def test_typecast_unary(self, fundat):

       w = fundat(self.backend)
       assert tn.allclose(w.wrappedfun(*w.args), w.out)




# --- Tensor unary elementwise operator ------------------------------------- #  

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTensorElemwiseUnary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Value methods --- #

   @pytest.mark.parametrize("indnames, shape, pos", [
      ["ijk", (2,3,4), (((1,0,1),), ((0,2,0),), ((2,1,3),))], 
   ])
   def test_put(self, indnames, shape, pos):

       backend = backends.get(self.backend)
       vals    = backend.randn((len(pos),), seed=1)

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.put(w.tensor, pos, vals)
       ans = ar.put(w.array,  pos, vals)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape, pos", [
      ["ijk", (2,3,4), (((1,0,1),), ((0,2,0),), ((2,1,3),))], 
   ])
   def test_put_accumulate(self, indnames, shape, pos):

       backend = backends.get(self.backend)
       vals    = backend.randn((len(pos),), seed=1)

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.put(w.tensor, pos, vals, accumulate=True)
       ans = ar.put(w.array,  pos, vals, accumulate=True)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape, minval, maxval", [
      ["ijk", (2,3,4), 0, 1], 
   ])
   def test_clip(self, indnames, shape, minval, maxval):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.clip(w.tensor, minval, maxval)
       ans = ar.clip(w.array, minval, maxval)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape, inds", [
      ["ijk", (2,3,4), None], 
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_flip(self, indnames, shape, inds):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       if   inds is None:
            out = tn.flip(w.tensor)
            ans = ar.flip(w.array)
            ans = tn.TensorGen(ans, w.inds)

       else:
            out = tn.flip(w.tensor, inds)
            ans = ar.flip(w.array,  w.inds.axes(*inds))
            ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape, ind", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "j"],
   ])
   def test_cumsum(self, indnames, shape, ind):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       if   ind is None:
            out = tn.cumsum(w.tensor)
            ans = ar.reshape(ar.cumsum(w.array), w.shape)
            ans = tn.TensorGen(ans, w.inds)

       else:
            out = tn.cumsum(w.tensor, ind)
            ans = ar.cumsum(w.array,  w.inds.axes(ind)[0])
            ans = tn.TensorGen(ans,   w.inds)

       assert tn.allclose(out, ans)


   # --- Element access --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_getitem(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       def elem(pos):
           return tn.TensorGen(w.array[pos])

       for pos in itertools.product(*map(range, shape)):
           assert w.tensor[pos] == elem(pos)


   # --- Extracting info --- #

   @pytest.mark.parametrize("dtype, iscomplex", [
      ["int64",      False], 
      ["float64",    False], 
      ["complex128", True],
   ])
   def test_iscomplex(self, dtype, iscomplex):

       w = data.tensor_dat(data.randn)(
              self.backend, "ijk", (2,3,4), dtype=dtype
           )

       assert tn.iscomplex(w.tensor) == iscomplex  


   # --- Standard math --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_floor(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, dtype="float64"
           )

       out = tn.floor(w.tensor)
       ans = ar.floor(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_neg(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = -w.tensor
       ans = ar.neg(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_sign(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.sign(w.tensor)
       ans = ar.sign(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_conj(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.conj(w.tensor)
       ans = ar.conj(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_real(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.real(w.tensor)
       ans = ar.real(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_imag(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.imag(w.tensor)
       ans = ar.imag(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_absolute(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.absolute(w.tensor)
       ans = ar.absolute(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_sqrt(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.sqrt(w.tensor)
       ans = ar.sqrt(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_log(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.log(w.tensor)
       ans = ar.log(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_exp(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.exp(w.tensor)
       ans = ar.exp(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_sin(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.sin(w.tensor)
       ans = ar.sin(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_cos(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.cos(w.tensor)
       ans = ar.cos(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_tan(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.tan(w.tensor)
       ans = ar.tan(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_arcsin(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.arcsin(w.tensor)
       ans = ar.arcsin(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_arccos(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.arccos(w.tensor)
       ans = ar.arccos(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_arctan(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.arctan(w.tensor)
       ans = ar.arctan(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_sinh(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.sinh(w.tensor)
       ans = ar.sinh(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_cosh(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.cosh(w.tensor)
       ans = ar.cosh(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_tanh(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.tanh(w.tensor)
       ans = ar.tanh(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_arcsinh(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.arcsinh(w.tensor)
       ans = ar.arcsinh(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_arccosh(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.arccosh(w.tensor)
       ans = ar.arccosh(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_arctanh(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tn.arctanh(w.tensor)
       ans = ar.arctanh(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)


   # --- Linear algebra --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ij", (4,4)],
   ])
   def test_expm(self, indnames, shape):

       w = data.tensor_dat(data.randn)(
              self.backend, indnames, shape
           )

       out = tnu.expm(w.tensor)
       ans = ar.expm(w.array)
       ans = tn.TensorGen(ans, w.inds)

       assert tn.allclose(out, ans)   




