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
import tadpole.linalg   as la

import tests.linalg.fakes as fake
import tests.linalg.data  as data

from tests.common import (
   available_backends,
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor linalg properties                                               ###
###                                                                         ###
###############################################################################


# --- Linear algebra properties --------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestLinalgProperties:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
   ])
   @pytest.mark.parametrize("order", [
      None, "fro", "nuc", 1, 2, -1, -2, 
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_norm(self, property_input, order, alignment):

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       out = la.norm(w.tensor, **inds, order=order)
       ans = ar.norm(w.matrix, order=order)
       ans = tn.TensorGen(ans) 

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_trace(self, property_input, alignment):

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       out = la.trace(w.tensor, **inds)
       ans = ar.trace(w.matrix)
       ans = tn.TensorGen(ans) 

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("property_input", [
      data.property_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_det(self, property_input, alignment):

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       out = la.det(w.tensor, **inds)
       ans = ar.det(w.matrix)
       ans = tn.TensorGen(ans) 

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("property_input", [
      data.property_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_inv(self, property_input, alignment):

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       out = la.inv(w.tensor, **inds)
       ans = ar.inv(w.matrix)
       ans = ar.reshape(ans,   (*w.lshape, *w.rshape))
       ans = tn.TensorGen(ans, (*w.linds,  *w.rinds)) 
       ans = tn.transpose(ans, *w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_tril(self, property_input, alignment):

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       out = la.tril(w.tensor, **inds)
       ans = ar.tril(w.matrix)
       ans = ar.reshape(ans,   (*w.lshape, *w.rshape))
       ans = tn.TensorGen(ans, (*w.linds,  *w.rinds)) 
       ans = tn.transpose(ans, *w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_triu(self, property_input, alignment):

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       out = la.triu(w.tensor, **inds)
       ans = ar.triu(w.matrix)
       ans = ar.reshape(ans,   (*w.lshape, *w.rshape))
       ans = tn.TensorGen(ans, (*w.linds,  *w.rinds)) 
       ans = tn.transpose(ans, *w.inds)

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_diag(self, property_input, alignment):

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend
           )

       inds = {
               "l":  {"linds": w.linds}, 
               "r":  {"rinds": w.rinds}, 
               "lr": {"linds": w.linds, "rinds": w.rinds},
              }[alignment]

       i = IndexGen("l", min(w.lsize, w.rsize))

       out = la.diag(w.tensor, (i,), **inds)
       ans = ar.diag(w.matrix)
       ans = tn.TensorGen(ans, (i,)) 

       assert tn.allclose(out, ans)




