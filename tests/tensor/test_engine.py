#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.tensor.elemwise_unary  as tnu
import tadpole.tensor.elemwise_binary as tnb
import tadpole.tensor.engine          as tne 

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
###  Train of tensor data/metadata                                          ###
###                                                                         ###
###############################################################################


# --- Train of tensor data/metadata ----------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTrainTensorData:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk"              ], [(2,3,4)]],
      [["ijk", "mn"        ], [(2,3,4), (4,5)]],
      [["ijk", "mn", "abcd"], [(2,3,4), (4,5), (3,5,6,7)]],
   ])
   def test_attach(self, indnames, shapes): 

       w   = data.train_dat(self.backend, indnames, shapes)       
       out = tne.TrainTensorData()

       for array, inds in zip(w.arrays, w.inds):
           out = out.attach(array, inds)

       assert out == w.train


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk"              ], [(2,3,4)]],
      [["ijk", "mn"        ], [(2,3,4), (4,5)]],
      [["ijk", "mn", "abcd"], [(2,3,4), (4,5), (3,5,6,7)]],
   ])
   def test_data(self, indnames, shapes): 

       w = data.train_dat(self.backend, indnames, shapes)       

       assert tuple(w.train.data()) == tuple(w.arrays)


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk"              ], [(2,3,4)]],
      [["ijk", "mn"        ], [(2,3,4), (4,5)]],
      [["ijk", "mn", "abcd"], [(2,3,4), (4,5), (3,5,6,7)]],
   ])
   def test_inds(self, indnames, shapes): 

       w = data.train_dat(self.backend, indnames, shapes)       

       assert tuple(w.train.inds()) == tuple(w.inds)


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk"              ], [(2,3,4)]],
      [["ijk", "mn"        ], [(2,3,4), (4,5)]],
      [["ijk", "mn", "abcd"], [(2,3,4), (4,5), (3,5,6,7)]],
   ])
   def test_size(self, indnames, shapes): 

       w = data.train_dat(self.backend, indnames, shapes)       

       assert w.train.size() == len(shapes)




###############################################################################
###                                                                         ###
###  Engine for unary operations                                            ###
###                                                                         ###
###############################################################################


# --- Unary engine ---------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestEngineUnary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk"], [(2,3,4)]],
   ])
   def test_attach(self, indnames, shapes): 

       w = data.train_dat(self.backend, indnames, shapes)

       ans = tne.EngineUnary(tnu.TensorElemwiseUnary, w.train)
       out = tne.EngineUnary(tnu.TensorElemwiseUnary)

       for array, inds in zip(w.arrays, w.inds):
           out = out.attach(array, inds)

       assert out == ans


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk", "mn"], [(2,3,4), (4,5)]],
   ])
   def test_attach_fail(self, indnames, shapes): 

       w = data.train_dat(self.backend, indnames, shapes)

       ans = tne.EngineUnary(tnu.TensorElemwiseUnary, w.train)
       out = tne.EngineUnary(tnu.TensorElemwiseUnary)

       try:
           for array, inds in zip(w.arrays, w.inds):
               out = out.attach(array, inds)

       except tne.TooManyArgsError:
           assert True
       else:
           assert False




###############################################################################
###                                                                         ###
###  Engine for elementwise operations                                      ###
###                                                                         ###
###############################################################################


# --- Elementwise engine ---------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestEngineElemwise:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk", "mn"], [(2,3,4), (4,5)]],
   ])
   def test_attach(self, indnames, shapes): 

       w = data.train_dat(self.backend, indnames, shapes)

       ans = tne.EngineElemwise(tnb.TensorElemwiseBinary, 2, w.train)
       out = tne.EngineElemwise(tnb.TensorElemwiseBinary, 2)

       for array, inds in zip(w.arrays, w.inds):
           out = out.attach(array, inds)

       assert out == ans


   @pytest.mark.parametrize("indnames, shapes", [
      [["ijk", "mn", "abcd"], [(2,3,4), (4,5), (3,5,6,7)]]
   ])
   def test_attach_fail(self, indnames, shapes): 

       w = data.train_dat(self.backend, indnames, shapes)

       ans = tne.EngineElemwise(tnb.TensorElemwiseBinary, 2, w.train)
       out = tne.EngineElemwise(tnb.TensorElemwiseBinary, 2)

       try:
           for array, inds in zip(w.arrays, w.inds):
               out = out.attach(array, inds)

       except tne.TooManyArgsError:
           assert True
       else:
           assert False















































