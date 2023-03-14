#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools
import numpy as np

from tests.common import options

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util             as util
import tadpole.array.core       as core
import tadpole.array.grad       as grad
import tadpole.array.backends   as backends
import tadpole.array.function   as function
import tadpole.array.operations as op

import tadpole.array as td




###############################################################################
###                                                                         ###
###  Array creation functions                                               ###
###                                                                         ###
###############################################################################


# --- Array generators ------------------------------------------------------ #

class TestArrayGenerators:

   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001,
   ])
   def test_units(self, backend, basisdat):

       x    = basisdat(backend)
       opts = options(dtype=x.dtype, backend=backend)

       ans = list(x.arrays)
       out = list(core.units(x.shape, **opts))

       assert out == ans


   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001, 
      data.basis_complex_dat_001,
   ])
   def test_basis(self, backend, basisdat):

       x    = basisdat(backend)
       opts = options(dtype=x.dtype, backend=backend)

       ans = list(x.arrays)
       out = list(core.basis(x.shape, **opts))

       assert out == ans




# --- Array factories (from data) ------------------------------------------- #

class TestArraysFromData:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])
   def test_fromfun(self, backend, shape, dtype):
       
       x1 = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=1
            )
       x2 = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=2
            )   
       ans = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=3
            ) 

       opts = options(dtype=dtype, backend=backend)

       fun = fake.Fun(ans.data, ans.backend, x1.data, x2.data)
       out = core.fromfun(fun, x1.data, x2.data, **opts)

       assert out == ans.array


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])
   def test_asarray(self, backend, shape, dtype):

       x = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=1
           )

       opts = options(dtype=dtype, backend=backend)
       out  = core.asarray(x.data, **opts) 

       assert out == x.array




# --- Array factories (from shape) ------------------------------------------ #

class TestArraysFromShape:

   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001,
   ])
   def test_unit(self, backend, basisdat):

       x    = basisdat(backend)
       opts = options(dtype=x.dtype, backend=backend)

       for idx, ans in zip(x.idxs, x.arrays):

           out = core.unit(x.shape, idx, **opts)
           assert out == ans


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.zeros_dat_001,
   ])
   def test_zeros(self, backend, sampledat):

       x    = sampledat(backend)
       opts = options(dtype=x.dtype, backend=backend)

       out = core.zeros(x.shape, **opts)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.ones_dat_001,
   ])
   def test_ones(self, backend, sampledat):

       x    = sampledat(backend)
       opts = options(dtype=x.dtype, backend=backend)

       out = core.ones(x.shape, **opts)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.rand_real_dat_001,
      data.rand_complex_dat_001,
   ])
   def test_rand(self, backend, sampledat):

       seed = 1
       x    = sampledat(backend, seed=seed)
       opts = options(dtype=x.dtype, seed=seed, backend=backend)

       out = core.rand(x.shape, **opts)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.randn_real_dat_001,
      data.randn_complex_dat_001,
   ])
   def test_randn(self, backend, sampledat):

       seed = 1
       x    = sampledat(backend, seed=seed)
       opts = options(dtype=x.dtype, seed=seed, backend=backend)

       out = core.randn(x.shape, **opts)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.randuniform_real_dat_001,
      data.randuniform_complex_dat_001,
   ])
   def test_randuniform(self, backend, sampledat):

       seed = 1
       x    = sampledat(backend, seed=seed)
       opts = options(dtype=x.dtype, seed=seed, backend=backend)

       out = core.randuniform(
                x.shape, x.opts["boundaries"], **opts
             )

       assert out == x.array




###############################################################################
###                                                                         ###
###  Array space                                                            ###
###                                                                         ###
###############################################################################


# --- ArraySpace ------------------------------------------------------------ #

class TestArraySpace:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])  
   def test_dtype(self, backend, shape, dtype):

       w = data.array_space_dat(
              backend, shape, dtype
           )

       assert w.space.dtype == dtype


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])  
   def test_size(self, backend, shape, dtype):

       w = data.array_space_dat(
              backend, shape, dtype
           )

       assert w.space.size == np.prod(shape) 


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])  
   def test_ndim(self, backend, shape, dtype):

       w = data.array_space_dat(
              backend, shape, dtype
           )

       assert w.space.ndim == len(shape)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])  
   def test_shape(self, backend, shape, dtype):

       w = data.array_space_dat(
              backend, shape, dtype
           )

       assert w.space.shape == shape




###############################################################################
###                                                                         ###
###  Definition of array.                                                   ###
###                                                                         ###
###############################################################################


# --- Array ----------------------------------------------------------------- #

class TestArray:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shapes",  [
      [(2,3,4)], 
      [(2,3,4), (4,5)],
   ])
   def test_pluginto(self, backend, shapes):

       array_dat = data.array_dat(data.randn)

       ws  = [array_dat(backend, shape) for shape in shapes] 
       seq = [(w.backend, w.data)       for w     in ws]

       fun     = fake.Fun(None)
       ans     = function.TransformCall(fun, util.Sequence(seq))
       funcall = function.TransformCall(fun)

       for w in ws:
           funcall = w.array.pluginto(funcall)
        
       assert funcall == ans


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_addto_zero(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       out = w.array.addto(grad.ZeroGrad())
       assert out is w.array


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_addto_dense(self, backend, shape):

       x = data.array_dat(data.randn)(backend, shape, seed=1)
       y = data.array_dat(data.randn)(backend, shape, seed=2)

       out = x.array.addto(y.array)
       assert td.allclose(out, x.array + y.array)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("graddat", [
      data.sparse_grad_dat_001,
      data.sparse_grad_dat_002,
   ])
   def test_addto_sparse(self, backend, graddat):

       y = graddat(backend)
       x = data.array_dat(data.randn)(backend, y.grad.shape)

       out = x.array.addto(y.grad)
       assert td.allclose(out, x.array + y.dense)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_copy(self, backend, shape):

       w   = data.array_dat(data.randn)(backend, shape)   
       out = w.array.copy()

       assert out == w.array
       assert out is not w.array
      

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   [
      "int64", 
      "float64", 
      "complex128",
   ])
   def test_space(self, backend, shape, dtype):

       w = data.array_dat(data.randn)(
              backend, shape, dtype=dtype)     

       space = core.ArraySpace(w.backend, shape, dtype)  
       assert w.array.space() == space


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("dtype",   [
      "int64", 
      "float64", 
      "complex128",
   ])
   def test_dtype(self, backend, dtype):

       w = data.array_dat(data.randn)(
              backend, (2,3,4), dtype=dtype)

       assert w.array.dtype == dtype 


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_size(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)
       assert w.array.size == np.prod(shape)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_ndim(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)
       assert w.array.ndim == len(shape)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_shape(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)
       assert w.array.shape == shape


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_getitem(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       def elem(idx):
           return core.asarray(w.data[idx], backend=w.backend)

       for idx in itertools.product(*map(range, shape)):
           assert w.array[idx] == elem(idx)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_item(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       for idx in itertools.product(*map(range, shape)):
           assert w.array.item(*idx) == w.data[idx]


   @pytest.mark.parametrize("backend", ["numpy"])
   def test_item_zerodim(self, backend):

       w = data.array_dat(data.randn)(backend, (1,))
       assert w.array.item() == w.data[0]



