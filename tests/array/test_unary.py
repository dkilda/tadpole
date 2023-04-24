#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools

import numpy as np
import scipy

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array          as ar
import tadpole.array.space    as sp
import tadpole.array.unary    as unary
import tadpole.array.binary   as binary
import tadpole.array.nary     as nary
import tadpole.array.backends as backends

from tests.common import (
   options,
   available_backends,
)




###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Type cast for unary functions ----------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTypeCast:

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
   def test_typecast(self, fundat):

       w = fundat(self.backend)
       assert ar.allclose(w.wrappedfun(*w.args), w.out)




###############################################################################
###                                                                         ###
###  Definition of Unary Array (supports unary operations)                  ###
###                                                                         ###
###############################################################################


# --- Unary Array ----------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestArray:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend
   

   # --- Array factory --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_asarray(self, shape, dtype):

       x = data.array_dat(data.randn)(
              self.backend, shape, dtype=dtype, seed=1
           )

       opts = options(dtype=dtype, backend=self.backend)
       out  = unary.asarray(x.data, **opts) 

       assert out == x.array


   # --- Array methods --- #   

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_new(self, shape, dtype):

       x = data.array_dat(data.randn)(
              self.backend, shape, dtype=dtype, seed=1
           )
       y = data.array_dat(data.randn)(
              self.backend, shape, dtype=dtype, seed=2
           )

       assert x.array.new(y.data) == y.array


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_nary(self, shape, dtype):

       x = data.array_dat(data.randn)(
              self.backend, shape, dtype=dtype
           )

       assert x.array.nary() == nary.Array(x.backend, x.data)


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (5,4,6)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "float64"],
   ])
   def test_or_binary(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       assert w.arrays[0] | w.arrays[1] == w.narray

   
   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (5,4,6), (3,5,2)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "float64", "complex128"],
   ])
   def test_or_nary(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       x = w.arrays[0]
       y = w.arrays[1] | w.arrays[2]

       assert x | y == w.narray


   # --- Core methods --- #   

   @pytest.mark.parametrize("shape", [(2,3,4)])
   def test_copy(self, shape):

       w   = data.array_dat(data.randn)(self.backend, shape)
       out = w.array.copy()

       assert out == w.array
       assert out is not w.array
      

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", [
      "int64", 
      "float64", 
      "complex128",
   ])
   def test_space(self, shape, dtype):

       w = data.array_dat(data.randn)(
              self.backend, shape, dtype=dtype)     

       space = sp.ArraySpace(w.backend, shape, dtype)  
       assert w.array.space() == space


   # --- Data type methods --- #

   @pytest.mark.filterwarnings('ignore::RuntimeWarning')
   @pytest.mark.parametrize("dtype1", [
      "int64", 
      "float64", 
      "complex128",
   ])
   @pytest.mark.parametrize("dtype2", [
      "int64", 
      "float64", 
      "complex128",
   ])
   def test_astype(self, dtype1, dtype2):

       w = data.array_dat(data.randn)(
              self.backend, (2,3,4), dtype=dtype1)

       out = w.array.astype(dtype=dtype2)

       assert out.dtype == dtype2 


   @pytest.mark.parametrize("dtype", [
      "int64", 
      "float64", 
      "complex128",
   ])
   def test_dtype(self, dtype):

       w = data.array_dat(data.randn)(
              self.backend, (2,3,4), dtype=dtype)

       assert w.array.dtype == dtype 


   @pytest.mark.parametrize("dtype, iscomplex", [
      ["int64",      False], 
      ["float64",    False], 
      ["complex128", True],
   ])
   def test_iscomplex(self, dtype, iscomplex):

       w = data.array_dat(data.randn)(
              self.backend, (2,3,4), dtype=dtype)

       assert ar.iscomplex(w.array) == iscomplex 


   # --- Element access --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   def test_getitem(self, shape):

       w = data.array_dat(data.randn)(self.backend, shape)

       def elem(idx):
           return unary.asarray(w.data[idx], backend=w.backend)

       for idx in itertools.product(*map(range, shape)):
           assert w.array[idx] == elem(idx)


   # --- Shape methods --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   def test_size(self, shape):

       w = data.array_dat(data.randn)(self.backend, shape)
       assert w.array.size == np.prod(shape)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   def test_ndim(self, shape):

       w = data.array_dat(data.randn)(self.backend, shape)
       assert w.array.ndim == len(shape)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   def test_shape(self, shape):

       w = data.array_dat(data.randn)(self.backend, shape)
       assert w.array.shape == shape


   @pytest.mark.parametrize("shape1, shape2", [
      [(2,3,4), (3,8)],
   ])
   def test_reshape(self, shape1, shape2):

       w = data.array_dat(data.randn)(self.backend, shape1)

       out = ar.reshape(w.array, shape2)
       ans = np.reshape(w.data,  shape2)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axes", [
      [(2,3,4), (2,0,1)],
   ])
   def test_transpose(self, shape, axes):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.transpose(w.array, axes)
       ans = np.transpose(w.data,  axes)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, source, destination", [
      [(2,3,4), 2, 0],
   ])
   def test_moveaxis(self, shape, source, destination):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.moveaxis(w.array, source, destination)
       ans = np.moveaxis(w.data,  source, destination)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4),     None],
      [(2,1,3,4),   None],
      [(2,1,3,4,1), None],
      [(2,1,3,4,1),  1],
      [(2,1,3,4,1), -1],
   ])
   def test_squeeze(self, shape, axis):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.squeeze(w.array, **options(axis=axis))
       ans = np.squeeze(w.data,  **options(axis=axis))
       ans = unary.asarray(ans,  **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), 1],
      [(2,3,4), (1,4)],
   ])
   def test_unsqueeze(self, shape, axis):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.unsqueeze(w.array,  axis)
       ans = np.expand_dims(w.data, axis)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
      [(2,3,4), (0,1)],
      [(2,3,4), (0,2)],
   ])
   @pytest.mark.parametrize("dtype", [
      "complex64",
      "complex128",
   ])
   def test_sumover(self, shape, axis, dtype):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.sumover(w.array, **options(axis=axis, dtype=dtype))
       ans = np.sum(w.data,      **options(axis=axis, dtype=dtype))
       ans = unary.asarray(ans,  **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
   ])
   @pytest.mark.parametrize("dtype", [
      "complex64",
      "complex128",
   ])
   def test_cumsum(self, shape, axis, dtype):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.cumsum(w.array, **options(axis=axis, dtype=dtype))
       ans = np.cumsum(w.data,  **options(axis=axis, dtype=dtype))
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape1, shape2", [
      [(2,3,4), (5,2,3,4)],
      [(2,3,4), (1,2,3,4)],
   ])
   def test_broadcast_to(self, shape1, shape2):

       w = data.array_dat(data.randn)(self.backend, shape1)

       out = ar.broadcast_to(w.array, shape2)
       ans = np.broadcast_to(w.data,  shape2)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   # --- Value methods --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   def test_item(self, shape):

       w = data.array_dat(data.randn)(self.backend, shape)

       for idx in itertools.product(*map(range, shape)):
           assert w.array.item(*idx) == w.data[idx]


   def test_item_zerodim(self):

       w = data.array_dat(data.randn)(self.backend, (1,))
       assert w.array.item() == w.data[0]


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
      [(2,3,4), (0,1)],
      [(2,3,4), (0,2)],
   ])
   @pytest.mark.parametrize("dtype", [
      "int64",
      "float64",
   ])
   def test_allof(self, shape, axis, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.allof(w.array,  **options(axis=axis))
       ans = np.all(w.data,     **options(axis=axis))
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
      [(2,3,4), (0,1)],
      [(2,3,4), (0,2)],
   ])
   @pytest.mark.parametrize("dtype", [
      "int64",
   ])
   def test_anyof(self, shape, axis, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.anyof(w.array,  **options(axis=axis))
       ans = np.any(w.data,     **options(axis=axis))
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
      [(2,3,4), (0,1)],
      [(2,3,4), (0,2)],
   ])
   def test_amax(self, shape, axis):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.amax(w.array,   **options(axis=axis))
       ans = np.amax(w.data,    **options(axis=axis))
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
      [(2,3,4), (0,1)],
      [(2,3,4), (0,2)],
   ])
   def test_amin(self, shape, axis):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.amin(w.array,   **options(axis=axis))
       ans = np.amin(w.data,    **options(axis=axis))
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, dtype", [
      [(2,3,4), "float64"],
      [(2,3,4), "complex128"],
   ])
   def test_sign(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.sign(w.array)
       ans = np.sign(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, dtype", [
      [(2,3,4), "float64"],
      [(2,3,4), "complex128"],
   ])
   def test_absolute(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.absolute(w.array)
       ans = np.absolute(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
      [(2,3,4), (0,1)],
      [(2,3,4), (0,2)],
   ])
   def test_flip(self, shape, axis):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.flip(w.array,   **options(axis=axis))
       ans = np.flip(w.data,    **options(axis=axis))
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, dtype, minval, maxval", [
      [(2,3,4), "float64",    0, 1],
      [(2,3,4), "complex128", 0, 1],
   ])
   def test_clip(self, shape, dtype, minval, maxval):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.clip(w.array, minval, maxval)
       ans = np.clip(w.data, minval, maxval)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
      [(2,3,4), (0,1)],
      [(2,3,4), (0,2)],
   ])
   def test_count_nonzero(self, shape, axis):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.count_nonzero(w.array, **options(axis=axis))
       ans = np.count_nonzero(w.data,  **options(axis=axis))
       ans = unary.asarray(ans,        **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, axis", [
      [(2,3,4), None],
      [(2,3,4), 1],
   ])
   def test_argsort(self, shape, axis):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.argsort(w.array, **options(axis=axis))
       ans = np.argsort(w.data,  **options(axis=axis))
       ans = unary.asarray(ans,  **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape", [(4,), (5,5), (4,5)])
   def test_diag(self, shape):

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.diag(w.array)
       ans = np.diag(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, idxs, smartidxs", [
      [(2,3,4), [(1,0,2), (0,2,1), (1,0,3)], (((1,0,1),), ((0,2,0),), ((2,1,3),))], 
   ])
   def test_put(self, shape, idxs, smartidxs):

       np.random.seed(1)
       vals = np.random.randn(len(idxs))

       w = data.array_dat(data.randn)(self.backend, shape)

       out               = ar.put(w.array, idxs, vals)
       w.data[smartidxs] = vals

       ans = unary.asarray(w.data, **options(backend=self.backend))

       assert out == ans


   @pytest.mark.parametrize("shape, idxs, smartidxs", [
      [(2,3,4), [(1,0,2), (0,2,1), (1,0,3)], (((1,0,1),), ((0,2,0),), ((2,1,3),))], 
   ])
   def test_put_accumulate(self, shape, idxs, smartidxs):

       np.random.seed(1)
       vals = np.random.randn(len(idxs))

       w = data.array_dat(data.randn)(self.backend, shape)

       out = ar.put(w.array, idxs, vals, accumulate=True)

       np.add.at(w.data, smartidxs, vals) 
       ans = unary.asarray(w.data, **options(backend=self.backend))

       assert out == ans


   # --- Standard math --- #

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["float64"])
   def test_floor(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.floor(w.array) 
       ans = w.backend.floor(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_neg(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.neg(w.array) 
       ans = -w.data
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_conj(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.conj(w.array) 
       ans = np.conj(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_real(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.real(w.array) 
       ans = np.real(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_imag(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.imag(w.array) 
       ans = np.imag(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_sqrt(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.sqrt(w.array) 
       ans = np.sqrt(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_log(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.log(w.array) 
       ans = np.log(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_exp(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.exp(w.array) 
       ans = np.exp(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_sin(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.sin(w.array)
       ans = np.sin(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_cos(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.cos(w.array)
       ans = np.cos(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_tan(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.tan(w.array)
       ans = np.tan(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_arcsin(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.arcsin(w.array)
       ans = np.arcsin(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_arccos(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.arccos(w.array)
       ans = np.arccos(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_arctan(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.arctan(w.array)
       ans = np.arctan(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_sinh(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.sinh(w.array)
       ans = np.sinh(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_cosh(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.cosh(w.array)
       ans = np.cosh(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_tanh(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.tanh(w.array)
       ans = np.tanh(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_arcsinh(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.arcsinh(w.array)
       ans = np.arcsinh(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_arccosh(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.arccosh(w.array)
       ans = np.arccosh(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_arctanh(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.arctanh(w.array)
       ans = np.arctanh(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   # --- Linear algebra: decompositions --- #

   @pytest.mark.parametrize("shape", [(3,4), (4,3), (4,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_svd(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       U, S, V = ar.svd(w.array)

       S1 = np.linalg.svd(w.data, full_matrices=False, compute_uv=False)
       assert ar.allclose(S, S1)

       dimU = shape[-2]
       dimS = min(shape[-2], shape[-1])
       dimV = shape[-1]

       assert U.shape == (dimU, dimS)
       assert S.shape == (dimS,     )
       assert V.shape == (dimS, dimV)

       x = ar.dot(U, ar.dot(ar.diag(S), V)) 
       assert ar.allclose(x, w.array)


   @pytest.mark.parametrize("shape", [(4,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_eig(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       U, S, V = ar.eig(w.array)

       dimU = shape[-2]
       dimS = min(shape[-2], shape[-1])
       dimV = shape[-1]

       assert U.shape == (dimU, dimS)
       assert S.shape == (dimS,     )
       assert V.shape == (dimS, dimV)

       x = ar.dot(U, ar.dot(ar.diag(S), V)) 
       assert ar.allclose(x, w.array)


   @pytest.mark.parametrize("shape", [(4,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_eigh(self, shape, dtype):

       w      = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)
       warray = ar.add(w.array, ar.transpose(ar.conj(w.array), (1,0)))
       
       U, S, V = ar.eigh(warray)

       dimU = shape[-2]
       dimS = min(shape[-2], shape[-1])
       dimV = shape[-1]

       assert U.shape == (dimU, dimS)
       assert S.shape == (dimS,     )
       assert V.shape == (dimS, dimV)

       x = ar.dot(U, ar.dot(ar.diag(S), V)) 
       assert ar.allclose(x, warray)


   @pytest.mark.parametrize("shape", [(3,4), (4,3), (4,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_qr(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       Q,  R  = ar.qr(w.array)
       Q1, R1 = np.linalg.qr(w.data, mode='reduced')

       assert ar.allclose(Q, Q1)
       assert ar.allclose(R, R1)


   @pytest.mark.parametrize("shape", [(3,4), (4,3), (4,4)])
   @pytest.mark.parametrize("dtype", ["complex128", "float64"])
   def test_lq(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       L,  Q  = ar.lq(w.array)
       Q1, R1 = np.linalg.qr(np.transpose(np.conj(w.data)), mode='reduced')

       assert ar.allclose(L, np.transpose(np.conj(R1)))
       assert ar.allclose(Q, np.transpose(np.conj(Q1)))


   # --- Linear algebra: matrix exponential --- #

   @pytest.mark.parametrize("shape", [(4,4)]) 
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_expm(self, shape, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.expm(w.array)
       ans = scipy.linalg.expm(w.data)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   # --- Linear algebra: norm --- #

   @pytest.mark.parametrize("shape, axis, order", [
      [(2,3,4), None,  None],
      [(2,3,4), 1,     None],
      [(2,3,4), 1,     0],
      [(2,3,4), (0,1), None],
      [(2,3,4), (0,1), "fro"],
      [(2,3,4), (0,1), "nuc"],
      [(2,3,4), (0,1),  1],
      [(2,3,4), (0,1),  2],
      [(2,3,4), (0,1), -1],
      [(2,3,4), (0,1), -2],
   ])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_norm(self, shape, axis, order, dtype):

       w = data.array_dat(data.randn)(self.backend, shape, dtype=dtype)

       out = ar.norm(w.array,       **options(axis=axis, order=order))
       ans = np.linalg.norm(w.data, **options(axis=axis, ord=order))
       ans = unary.asarray(ans,     **options(backend=self.backend))

       assert ar.allclose(out, ans)




