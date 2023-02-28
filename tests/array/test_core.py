#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array.core     as tcore
import tadpole.array.function as tfunction




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
       ans     = tfunction.FunCall(fun, util.Sequence(seq))
       funcall = tfunction.FunCall(fun)

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

       space = tcore.ArraySpace(w.backend, shape, dtype)  
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
































