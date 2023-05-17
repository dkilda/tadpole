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
###  Tensor linalg solvers                                                  ###
###                                                                         ###
###############################################################################


# --- Linear algebra solvers ------------------------------------------------ #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestLinalgSolvers:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_solve(self, solver_input, alignment):

       w = data.solve_linalg_dat(solver_input)(
              data.randn, self.backend
           )
       inds = {
               "l":  {"linds": w.indsI}, 
               "r":  {"rinds": w.indsJ}, 
               "lr": {"linds": w.indsI, "rinds": w.indsJ},
              }[alignment]

       out = la.solve(w.tensorA, w.tensorB, **inds)
       ans = w.tensorX

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_trisolve(self, solver_input, alignment):

       w = data.trisolve_upper_linalg_dat(solver_input)(
              data.randn, self.backend
           )
       inds = {
               "l":  {"linds": w.indsI}, 
               "r":  {"rinds": w.indsJ}, 
               "lr": {"linds": w.indsI, "rinds": w.indsJ},
              }[alignment]

       out = la.trisolve(w.tensorA, w.tensorB, **inds)
       ans = w.tensorX

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_trisolve_lower(self, solver_input, alignment):

       w = data.trisolve_lower_linalg_dat(solver_input)(
              data.randn, self.backend
           )
       inds = {
               "l":  {"linds": w.indsI}, 
               "r":  {"rinds": w.indsJ}, 
               "lr": {"linds": w.indsI, "rinds": w.indsJ},
              }[alignment]

       out = la.trisolve(w.tensorA, w.tensorB, **inds, which="lower")
       ans = w.tensorX

       assert tn.allclose(out, ans)


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("alignment", [
      "l", "r", "lr", 
   ])
   def test_trisolve_upper(self, solver_input, alignment):

       w = data.trisolve_upper_linalg_dat(solver_input)(
              data.randn, self.backend
           )
       inds = {
               "l":  {"linds": w.indsI}, 
               "r":  {"rinds": w.indsJ}, 
               "lr": {"linds": w.indsI, "rinds": w.indsJ},
              }[alignment]

       out = la.trisolve(w.tensorA, w.tensorB, **inds, which="upper")
       ans = w.tensorX

       assert tn.allclose(out, ans)




