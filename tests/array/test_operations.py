#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools
import numpy as np

from tadpole.tests.common import options

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util             as util
import tadpole.array.backends   as backends
import tadpole.array.core       as core
import tadpole.array.function   as function
import tadpole.array.operations as op




###############################################################################
###                                                                         ###
###  Definitions of differentiable array operations                         ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

class TestUnaryOperations:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_getitem(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       for idx in itertools.product(*map(range, shape)):
           assert op.getitem(w.array, idx) == w.data[idx]


   @pytest.mark.parametrize("backend",     ["numpy"])
   @pytest.mark.parametrize("shape, idxs", [
      [(2,3,4), (((1,0,1),), ((0,2,0),), ((2,1,3),))], 
   ])
   def test_put(self, backend, shape, idxs):

       np.random.seed(1)
       vals = np.random.randn(len(idxs))

       w   = data.array_dat(data.randn)(backend, shape)
       out = op.put(w.array, idxs, vals)

       w.data[idxs] = vals
       ans = core.asarray(w.data, **options(backend=backend))

       assert out == ans


   @pytest.mark.parametrize("backend",     ["numpy"])
   @pytest.mark.parametrize("shape, idxs", [
      [(2,3,4), (((1,0,1),), ((0,2,0),), ((2,1,3),))], 
   ])
   def test_put_accumulate(self, backend, shape, idxs):

       np.random.seed(1)
       vals = np.random.randn(len(idxs))

       w   = data.array_dat(data.randn)(backend, shape)
       out = op.put(w.array, idxs, vals, accumulate=True)

       np.add.at(w.data, idxs, vals) 
       ans = core.asarray(w.data, **options(backend=backend))

       assert out == ans


   @pytest.mark.parametrize("backend",       ["numpy"])
   @pytest.mark.parametrize("shape, shape1", [
      [(2,3,4), (3,8)],
   ])
   def test_reshape(self, backend, shape, shape1):

       w = data.array_dat(data.randn)(backend, shape)

       out = op.reshape(w.array, shape1)
       ans = np.reshape(w.data,  shape1)
       ans = core.asarray(ans, **options(backend=backend))

       assert out == ans


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_neg(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       out = -w.array 
       ans = -w.data
       ans = core.asarray(ans, **options(backend=backend))

       assert core.allclose(out, ans)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_sin(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       out = op.sin(w.array)
       ans = np.sin(w.data)
       ans = core.asarray(ans, **options(backend=backend))

       assert core.allclose(out, ans)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_cos(self, backend, shape):

       w = data.array_dat(data.randn)(backend, shape)

       out = op.cos(w.array)
       ans = np.cos(w.data)
       ans = core.asarray(ans, **options(backend=backend))

       assert core.allclose(out, ans)





# --- Array operations: binary ---------------------------------------------- #

class TestBinaryOperations:

   @pytest.mark.parametrize("backend",        ["numpy"])
   @pytest.mark.parametrize("shape1, shape2", [
      [(2,3,4), (2,3,4)],
   ])
   def test_add(self, backend, shape1, shape2):

       x1 = data.array_dat(data.randn)(backend, shape1)
       x2 = data.array_dat(data.randn)(backend, shape2)

       out = x1.array + x2.array
       ans = x1.data  + x2.data
       ans = core.asarray(ans, **options(backend=backend))
       
       return core.allclose(out, ans)


   @pytest.mark.parametrize("backend",        ["numpy"])
   @pytest.mark.parametrize("shape1, shape2", [
      [(2,3,4), (2,3,4)],
   ])
   def test_sub(self, backend, shape1, shape2):

       x1 = data.array_dat(data.randn)(backend, shape1)
       x2 = data.array_dat(data.randn)(backend, shape2)

       out = x1.array - x2.array
       ans = x1.data  - x2.data
       ans = core.asarray(ans, **options(backend=backend))
       
       return core.allclose(out, ans)


   @pytest.mark.parametrize("backend",        ["numpy"])
   @pytest.mark.parametrize("shape1, shape2", [
      [(2,3,4), (2,3,4)],
   ])
   def test_mul(self, backend, shape1, shape2):

       x1 = data.array_dat(data.randn)(backend, shape1)
       x2 = data.array_dat(data.randn)(backend, shape2)

       out = x1.array * x2.array
       ans = x1.data  * x2.data
       ans = core.asarray(ans, **options(backend=backend))
       
       return core.allclose(out, ans)
       



# --- Array operations: nary ------------------------------------------------ #

class TestNaryOperations:

   @pytest.mark.parametrize("backend",          ["numpy"])
   @pytest.mark.parametrize("equation, shapes", [
      ["ijk,klm->ijlm", [(3,4,6), (6,2,5)]],
   ])
   def test_einsum(self, backend, equation, shapes):

       xs = []
       for shape in shapes:
           x = data.array_dat(data.randn)(backend, shape)
           xs.append(x)

       out = op.einsum(equation, *(x.array for x in xs))
       ans = np.einsum(equation, *(x.data  for x in xs))
       ans = core.asarray(ans, **options(backend=backend))

       assert core.allclose(out, ans) 


















