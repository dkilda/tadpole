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


# --- Type cast for binary functions ---------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTypeCast:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("fundat",  [
       data.binary_wrappedfun_dat_001,
       data.binary_wrappedfun_dat_002,
       data.binary_wrappedfun_dat_003,
       data.binary_wrappedfun_dat_004,
   ])
   def test_typecast(self, fundat):

       w = fundat(self.backend)
       assert ar.allclose(w.wrappedfun(*w.args), w.out)




###############################################################################
###                                                                         ###
###  Definition of Binary Array (supports binary operations)                ###
###                                                                         ###
###############################################################################


# --- Binary Array ---------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestArray:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend
   

   # --- Array methods --- #

   @pytest.mark.parametrize("shapes, newshape", [
      [[(2,3,4), (2,3,4)], (5,3,2)]
   ])
   @pytest.mark.parametrize("dtypes, newdtype", [
      [["complex128", "complex128"], "float64"],
   ])
   def test_new(self, shapes, newshape, dtypes, newdtype):

       w = data.narray_dat(data.randn)(
              self.backend, shapes, dtypes, seed=1
           )
       x = data.array_dat(data.randn)(
              self.backend, newshape, dtype=newdtype, seed=11
           )

       assert w.narray.new(x.data) == x.array


   @pytest.mark.parametrize("shapes", [[(2,3,4), (2,3,4)]])
   @pytest.mark.parametrize("dtypes", [["complex128", "complex128"]])
   def test_nary(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(
              self.backend, shapes, dtypes
           )

       assert w.narray.nary() == nary.Array(w.backend, *w.datas)
   

   @pytest.mark.parametrize("shapes1, shapes2", [
      [[(2,3,4), (5,4,6)], [(5,3,2), (3,4,2)]],
      [[(2,3,4), (5,4,6)], [(3,4,2)         ]],
   ])
   def test_or(self, shapes1, shapes2):

       w1 = data.narray_dat(data.randn)(self.backend, shapes1)
       w2 = data.narray_dat(data.randn)(self.backend, shapes2)

       ans = nary.Array(backends.get(self.backend), *w1.datas, *w2.datas)

       assert w1.narray | w2.narray == ans


   # --- Logical operations --- #

   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
      [[(2,3,4), (2,3,4)], 0],
   ])
   def test_allclose(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=-2.37+0.58j
           )

       if   nvals > 0:
            assert not ar.allclose(w.arrays[0], w.arrays[1]) 
       else:
            assert ar.allclose(w.arrays[0], w.arrays[1])

       assert ar.allclose(w.arrays[0], w.arrays[0])
       assert ar.allclose(w.arrays[0], ar.add(w.arrays[0], 1e-12))


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
      [[(2,3,4), (2,3,4)], 0],
   ])
   def test_allequal(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=1+0j
           )

       if   nvals > 0:
            assert not ar.allequal(w.arrays[0], w.arrays[1])
       else:
            assert ar.allequal(w.arrays[0], w.arrays[1])

       assert     ar.allequal(w.arrays[0], w.arrays[0])
       assert not ar.allequal(w.arrays[0], ar.add(w.arrays[0], 1e-12))


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_isclose(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=-2.37+0.58j
           )

       out = ar.isclose(w.arrays[0], w.arrays[1])
       ans = np.isclose(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans   


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_isequal(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=1+0j
           )

       out = ar.isequal(w.arrays[0], w.arrays[1])
       ans = np.equal(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_notequal(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=1+0j
           )

       out = ar.notequal(w.arrays[0], w.arrays[1])
       ans = np.not_equal(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_greater(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=1+0j
           )

       out = ar.greater(w.arrays[0], w.arrays[1])
       ans = w.datas[0] > w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_less(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=1+0j
           )

       out = ar.less(w.arrays[0], w.arrays[1])
       ans = w.datas[0] < w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_greater_equal(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=1+0j
           )

       out = ar.greater_equal(w.arrays[0], w.arrays[1])
       ans = w.datas[0] >= w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_less_equal(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, nvals=nvals, defaultval=1+0j
           )

       out = ar.less_equal(w.arrays[0], w.arrays[1])
       ans = w.datas[0] <= w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_logical_and(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, dtypes=["bool"]*len(shapes), 
              nvals=nvals, defaultval=False
           )

       out = ar.logical_and(w.arrays[0], w.arrays[1])
       ans = np.logical_and(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 


   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   def test_logical_or(self, shapes, nvals):

       w = data.narray_dat(data.randn_pos)(
              self.backend, shapes, dtypes=["bool"]*len(shapes), 
              nvals=nvals, defaultval=False
           )

       out = ar.logical_or(w.arrays[0], w.arrays[1])
       ans = np.logical_or(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))    

       assert out == ans 

   
   # --- Elementwise binary algebra --- #

   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (2,3,4)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_add(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.add(w.arrays[0], w.arrays[1])
       ans = w.datas[0] + w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (2,3,4)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_sub(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.sub(w.arrays[0], w.arrays[1])
       ans = w.datas[0] - w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (2,3,4)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_mul(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.mul(w.arrays[0], w.arrays[1])
       ans = w.datas[0] * w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (2,3,4)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_div(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.div(w.arrays[0], w.arrays[1])
       ans = w.datas[0] / w.datas[1]
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("sampledat", [
      data.randuniform_int_dat_001,
      data.randuniform_real_dat_001,
   ])
   def test_mod(self, sampledat):

       x = sampledat(self.backend, boundaries=(1,11), seed=1)
       y = sampledat(self.backend, boundaries=(1,11), seed=2)

       out = ar.mod(x.array, y.array)
       ans = x.data % y.data
       ans = unary.asarray(
                ans, **options(backend=self.backend, dtype=ans.dtype)
             )

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("sampledat", [
      data.randuniform_int_dat_001,
   ])
   def test_floordiv(self, sampledat):

       x = sampledat(self.backend, boundaries=(1,11), seed=1)
       y = sampledat(self.backend, boundaries=(1,11), seed=2)

       out = ar.floordiv(x.array, y.array)
       ans = x.data // y.data
       ans = unary.asarray(
                ans, **options(backend=self.backend, dtype=ans.dtype)
             )

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (2,3,4)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_power(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.power(w.arrays[0], w.arrays[1])
       ans = np.power(w.datas[0], w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   # --- Linear algebra: products --- #

   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (6,4,5)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_dot(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.dot(w.arrays[0], w.arrays[1])
       ans = np.dot(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (2,5,6)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_dot_fail(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       try:
           out = ar.dot(w.arrays[0], w.arrays[1])
       except ValueError:
           assert True
       else:
           assert False


   @pytest.mark.parametrize("shapes", [
      [(2,3,4), (2,5,6)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_kron(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.kron(w.arrays[0], w.arrays[1])
       ans = np.kron(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   # --- Linear algebra --- #

   @pytest.mark.parametrize("shapes", [
      [(4,4), (4,5)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_solve(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.solve(w.arrays[0], w.arrays[1])
       ans = np.linalg.solve(w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("shapes", [
      [(4,4), (4,5)],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   @pytest.mark.parametrize("which, lower", [
      ["lower", True],
      ["upper", False],
   ])
   def test_trisolve(self, shapes, dtypes, which, lower):

       w = data.narray_dat(data.randn)(self.backend, shapes, dtypes)

       out = ar.trisolve(w.arrays[0], w.arrays[1], which=which)
       ans = scipy.linalg.solve_triangular(w.datas[0], w.datas[1], lower=lower)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)






















