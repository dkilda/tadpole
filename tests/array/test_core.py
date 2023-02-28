#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array.backends as backends
import tadpole.array.core     as core
import tadpole.array.function as function




###############################################################################
###                                                                         ###
###  Array creation functions                                               ###
###                                                                         ###
###############################################################################


# --- Array factories ------------------------------------------------------- #

class TestArrayFactories:

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

       fun = fake.Fun(ans.data, ans.backend, x1.data, x2.data)
       out = core.fromfun(fun, backend, x1.data, x2.data, dtype=dtype)

       assert out == ans.array


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])
   def test_asarray(self, backend, shape, dtype):

       x = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=1
           )

       assert core.asarray(backend, x.data, dtype=dtype) == x.array


   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001,
   ])
   def test_unit(self, backend, basisdat):

       x = basisdat(backend)

       for idx, ans in zip(x.idxs, x.arrays):

           out = core.unit(backend, x.shape, idx, dtype=x.dtype)
           assert out == ans


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.zeros_dat_001,
   ])
   def test_zeros(self, backend, sampledat):

       x   = sampledat(backend)
       out = core.zeros(backend, x.shape, dtype=x.dtype)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.ones_dat_001,
   ])
   def test_ones(self, backend, sampledat):

       x   = sampledat(backend)
       out = core.ones(backend, x.shape, dtype=x.dtype)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.rand_real_dat_001,
      data.rand_complex_dat_001,
   ])
   def test_rand(self, backend, sampledat):

       x   = sampledat(backend, seed=1)
       out = core.rand(backend, x.shape, dtype=x.dtype, seed=1)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.randn_real_dat_001,
      data.randn_complex_dat_001,
   ])
   def test_randn(self, backend, sampledat):

       x   = sampledat(backend, seed=1)
       out = core.randn(backend, x.shape, dtype=x.dtype, seed=1)

       assert out == x.array


   @pytest.mark.parametrize("backend",   ["numpy"])
   @pytest.mark.parametrize("sampledat", [
      data.randuniform_real_dat_001,
      data.randuniform_complex_dat_001,
   ])
   def test_randuniform(self, backend, sampledat):

       x   = sampledat(backend, seed=1)
       out = core.randuniform(
                backend, x.shape, x.opts["boundaries"], dtype=x.dtype, seed=1
             )

       assert out == x.array




# --- Array generators ------------------------------------------------------ #

class TestArrayGenerators:

   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001,
   ])
   def test_units(self, backend, basisdat):

       x = basisdat(backend)

       ans = list(x.arrays)
       out = list(core.units(backend, x.shape, dtype=x.dtype))

       assert out == ans


   @pytest.mark.parametrize("backend",  ["numpy"])
   @pytest.mark.parametrize("basisdat", [
      data.basis_real_dat_001, 
      data.basis_complex_dat_001,
   ])
   def test_basis(self, backend, basisdat):

       x = basisdat(backend)

       ans = list(x.arrays)
       out = list(core.basis(backend, x.shape, dtype=x.dtype))

       assert out == ans




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
   def test_apply(self, backend, shape, dtype):

       w = data.array_space_dat(
              backend, shape, dtype
           )
       x1 = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=1
            )
       x2 = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=2
            )   
       out = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=3
            ) 

       fun = fake.Fun(out.data, out.backend, x1.data, x2.data)

       assert w.space.apply(fun, x1.data, x2.data) == out.array 

          
   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   @pytest.mark.parametrize("dtype",   ["complex128"])       
   def test_visit(self, backend, shape, dtype): 

       w = data.array_space_dat(
              backend, shape, dtype
           )
       x1 = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=1
            )
       x2 = data.array_dat(data.randn)(
              backend, shape, dtype=dtype, seed=2
            )   

       out = fake.Value()
       fun = fake.Fun(out, x1.backend, x1.data, x2.data)

       assert w.space.visit(fun, x1.data, x2.data) == out 


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
       seq = [(w.array, w.data)         for w     in ws]

       fun     = fake.Fun(None)
       ans     = function.FunCall(fun, util.Sequence(seq))
       funcall = function.FunCall(fun)

       for w in ws:
           funcall = w.array.pluginto(funcall)
        
       assert funcall == ans


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_radd(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       assert 0 + w.array is w.array


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_rmul(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       assert 1 * w.array is w.array
       

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

       for idx in itertools.product(*map(range, shape)):
           assert w.array[idx] == w.data[idx]




