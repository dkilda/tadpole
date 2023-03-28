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
)




###############################################################################
###                                                                         ###
###  Definition of Nary Array (supports nary operations)                    ###
###                                                                         ###
###############################################################################


# --- Nary Array ------------------------------------------------------------ #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestArray:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Array methods --- #

   @pytest.mark.parametrize("shapes, newshape", [
      [[(2,3,4), (2,3,4), (2,3,4)], (5,3,2)]
   ])
   @pytest.mark.parametrize("dtypes, newdtype", [
      [["complex128", "complex128", "complex128"], "float64"],
   ])
   def test_new(self, shapes, newshape, dtypes, newdtype):

       w = data.narray_dat(data.randn)(
              self.backend, shapes, dtypes, seed=1
           )
       x = data.array_dat(data.randn)(
              self.backend, newshape, dtype=newdtype, seed=11
           )

       assert w.narray.new(x.data) == x.array


   @pytest.mark.parametrize("shapes", [[(2,3,4), (2,3,4), (2,3,4)]])
   @pytest.mark.parametrize("dtypes", [["complex128", "complex128", "complex128"]])
   def test_nary(self, shapes, dtypes):

       w = data.narray_dat(data.randn)(
              self.backend, shapes, dtypes
           )

       assert w.narray.nary() is w.narray
   

   @pytest.mark.parametrize("shapes1, shapes2", [
      [[(2,3,4), (5,4,6), (5,3,2)], [(7,2,5), (3,4,2)]],
   ])
   def test_or(self, shapes1, shapes2):

       w1 = data.narray_dat(data.randn)(self.backend, shapes1)
       w2 = data.narray_dat(data.randn)(self.backend, shapes2)

       ans = nary.Array(backends.get(self.backend), *w1.datas, *w2.datas)

       assert w1.narray | w2.narray == ans


   # --- Value methods --- #

   @pytest.mark.parametrize("shapes, nvals", [
      [[(2,3,4), (2,3,4)], 5],
   ])
   @pytest.mark.parametrize("dtypes", [
      ["complex128", "complex128"],
   ])
   def test_where(self, shapes, nvals, dtypes):

       x = data.array_dat(data.randn_pos)(
              self.backend, shapes[0], dtype="bool", 
              nvals=nvals, defaultval=False
           )
       w = data.narray_dat(data.randn)(
              self.backend, shapes, dtypes
           )

       out = ar.where(x.array, w.arrays[0], w.arrays[1])
       ans = np.where(x.data,  w.datas[0],  w.datas[1])
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   # --- Linear algebra: products --- #

   @pytest.mark.parametrize("equation, shapes, dtypes", [
      ["ijk,klm->ijlm",     [(3,4,6), (6,2,5)           ], ["complex128"]*2],
      ["ijk,klm,mqlj->imq", [(3,4,6), (6,2,5), (5,7,2,4)], ["complex128"]*3],
   ])
   def test_einsum(self, equation, shapes, dtypes):

       w = data.narray_dat(data.randn)(
              self.backend, shapes, dtypes
           )

       out = ar.einsum(equation, *w.arrays)
       ans = np.einsum(equation, *w.datas)
       ans = unary.asarray(ans, **options(backend=self.backend))

       assert ar.allclose(out, ans)


   @pytest.mark.parametrize("equation, shapes, dtypes", [
      ["ijk,klm->ijlm",     [(3,4,6), (7,2,5)           ], ["complex128"]*2],
      ["ijk,klm,mqlj->imq", [(3,4,6), (6,2,5), (2,7,2,4)], ["complex128"]*3],
   ])
   def test_einsum_fail(self, equation, shapes, dtypes):

       w = data.narray_dat(data.randn)(
              self.backend, shapes, dtypes
           )

       try:
           out = ar.einsum(equation, *w.arrays)
       except ValueError:
           assert True
       else:
           assert False




